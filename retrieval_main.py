"""Train and evaluate MV-HFMD as a multi-view hotel retriever."""

import argparse
import hashlib
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets import (
    HotelBalancedBatchSampler,
    OpenHotelsFlatQueryDataset,
    OpenHotelsGalleryDataset,
    OpenHotelsQueryDataset,
    OpenHotelsRetrievalTrainDataset,
    build_openhotels_seen_validation,
    gallery_collate_fn,
    retrieval_collate_fn,
)
from engine.retrieval_evaluator import RetrievalEvaluator
from engine.retrieval_trainer import RetrievalTrainer, build_retrieval_scheduler
from loss import EasyPositiveLoss
from model import MultiImageHybrid


def _loader_kwargs(args):
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_checkpoint_metadata(payload, classes=None, config=None):
    """Reject incompatible label order or model settings when metadata is saved."""
    if not isinstance(payload, dict):
        return
    if classes is not None and "classes" in payload:
        if payload["classes"] != list(classes):
            raise ValueError("Checkpoint hotel classes do not match the dataset class order")
    if config is not None:
        saved_config = payload.get("config", {})
        for key, value in config.items():
            if key in saved_config and saved_config[key] != value:
                raise ValueError(
                    f"Checkpoint {key}={saved_config[key]!r} does not match {value!r}"
                )


def _load_checkpoint(model, checkpoint, logger, classes=None, config=None):
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    _validate_checkpoint_metadata(payload, classes, config)
    state_dict = payload.get("model", payload)
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    logger.info("Loaded checkpoint %s", path)
    return payload


def _subset_queries(dataset, limit, seed):
    if not limit or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, sorted(indices))


def _cap_gallery_per_hotel(dataset, limit):
    if not limit:
        return dataset
    counts = {}
    indices = []
    for index, row in enumerate(dataset.rows):
        hotel_id = str(row["hotel_id"])
        count = counts.get(hotel_id, 0)
        if count < limit:
            indices.append(index)
            counts[hotel_id] = count + 1
    return Subset(dataset, indices)


def _make_eval_loaders(gallery_dataset, query_dataset, args):
    kwargs = _loader_kwargs(args)
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=args.gallery_batch_size, shuffle=False,
        collate_fn=gallery_collate_fn, **kwargs,
    )
    query_loader = DataLoader(
        query_dataset, batch_size=args.query_batch_size, shuffle=False,
        collate_fn=retrieval_collate_fn, **kwargs,
    )
    return gallery_loader, query_loader


def _evaluation_kwargs(args, compute_image_metrics, compute_hotel_metrics=True):
    return {
        "ks": tuple(args.recall_at),
        "hotel_top_m": tuple(args.hotel_top_m),
        "hybrid_alpha": args.hybrid_alpha,
        "query_chunk_size": args.retrieval_query_chunk_size,
        "gallery_chunk_size": args.retrieval_gallery_chunk_size,
        "compute_image_metrics": compute_image_metrics,
        "compute_hotel_metrics": compute_hotel_metrics,
    }


def _best_validation_score(metrics):
    hotel = metrics["hotel"]
    aggregation = "max" if "max" in hotel else next(iter(hotel))
    modes = hotel[aggregation]
    mode = "hybrid" if "hybrid" in modes else "joint"
    return modes[mode]["recall@1"]


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _checkpoint_cache_tag(checkpoint):
    if not checkpoint:
        return "uninitialized"
    path = Path(checkpoint)
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _embedding_cache_tag(args, checkpoint):
    """Fingerprint every input that changes extracted evaluation embeddings."""

    data_dir = Path(args.data_dir).resolve()
    metadata = {}
    for filename in (
        "metadata_gallery.json",
        "metadata_test_room.json",
        "metadata_test_object.json",
    ):
        path = data_dir / filename
        stat = path.stat()
        metadata[filename] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    configuration = {
        "schema": 2,
        "checkpoint": _checkpoint_cache_tag(checkpoint),
        "data_dir": str(data_dir),
        "metadata": metadata,
        "architecture": args.architecture,
        "image_size": args.image_size,
        "model_max_views": args.model_max_views,
        "max_query_views": args.max_query_views,
        "test_min_views": args.test_min_views,
        "test_collections_per_group": args.test_collections_per_group,
        "max_test_queries": args.max_test_queries,
        "max_test_image_queries": args.max_test_image_queries,
        "skip_missing": args.skip_missing,
        "seed": args.seed,
    }
    encoded = json.dumps(configuration, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def build_model(args, num_classes, device, logger, checkpoint=None,
                return_checkpoint=False, classes=None):
    model = MultiImageHybrid(
        args.architecture,
        num_classes=num_classes,
        n=args.model_max_views,
        pretrained_weights=args.pretrained_weights and not checkpoint,
        image_size=args.image_size,
    )
    config = {
        "dataset": "openhotels",
        "architecture": args.architecture,
        "image_size": args.image_size,
        "model_max_views": args.model_max_views,
    }
    payload = _load_checkpoint(model, checkpoint, logger, classes, config)
    model = model.to(device)
    return (model, payload) if return_checkpoint else model


def run_official_evaluation(model, classes, args, device, logger, checkpoint_tag):
    logger.info("Building the full OpenHotels image gallery for retrieval evaluation")
    gallery_dataset = OpenHotelsGalleryDataset(
        args.data_dir, split="gallery", classes=classes, seed=args.seed,
        skip_missing=args.skip_missing, image_size=args.image_size,
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=args.gallery_batch_size, shuffle=False,
        collate_fn=gallery_collate_fn, **_loader_kwargs(args),
    )
    evaluator = RetrievalEvaluator(
        model=model, device=device, logger=logger,
        log_interval=args.eval_log_interval,
    )
    cache_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else None
    gallery_cache = (
        cache_dir / f"gallery-{checkpoint_tag}.pt" if cache_dir else None
    )
    gallery = evaluator.extract_gallery(
        gallery_loader, cache_path=gallery_cache,
        overwrite_cache=not args.reuse_embedding_cache,
    )

    all_metrics = {}
    for split in ("test_room", "test_object"):
        query_dataset = OpenHotelsQueryDataset(
            args.data_dir, split=split, classes=classes,
            min_views=args.test_min_views,
            max_views=args.max_query_views,
            collections_per_group=args.test_collections_per_group,
            seed=args.seed, skip_missing=args.skip_missing,
            image_size=args.image_size,
        )
        query_dataset = _subset_queries(
            query_dataset, args.max_test_queries, args.seed + 29,
        )
        query_loader = DataLoader(
            query_dataset, batch_size=args.query_batch_size, shuffle=False,
            collate_fn=retrieval_collate_fn, **_loader_kwargs(args),
        )
        query_cache = (
            cache_dir / f"{split}-{checkpoint_tag}.pt" if cache_dir else None
        )
        queries = evaluator.extract_queries(
            query_loader, cache_path=query_cache,
            overwrite_cache=not args.reuse_embedding_cache,
        )
        metrics = evaluator.evaluate_embeddings(
            queries["joint_embeddings"], queries["view_embeddings"],
            queries["labels"], gallery["embeddings"], gallery["labels"],
            query_view_mask=queries["view_mask"],
            **_evaluation_kwargs(args, compute_image_metrics=True),
        )

        # Preserve the official single-image OpenHotels protocol over every test
        # row. The grouped dataset intentionally selects at most four diverse
        # views and therefore cannot substitute for this metric.
        flat_dataset = OpenHotelsFlatQueryDataset(
            args.data_dir, split=split, classes=classes,
            skip_missing=args.skip_missing, image_size=args.image_size,
        )
        flat_dataset = _subset_queries(
            flat_dataset, args.max_test_image_queries, args.seed + 43,
        )
        flat_loader = DataLoader(
            flat_dataset, batch_size=args.gallery_batch_size, shuffle=False,
            collate_fn=gallery_collate_fn, **_loader_kwargs(args),
        )
        flat_cache = (
            cache_dir / f"{split}-single-{checkpoint_tag}.pt" if cache_dir else None
        )
        flat_queries = evaluator.extract_gallery(
            flat_loader, cache_path=flat_cache,
            overwrite_cache=not args.reuse_embedding_cache,
            deduplicate_paths=False,
        )
        single_metrics = evaluator.evaluate_embeddings(
            flat_queries["embeddings"], None, flat_queries["labels"],
            gallery["embeddings"], gallery["labels"],
            **_evaluation_kwargs(
                args, compute_image_metrics=True, compute_hotel_metrics=False,
            ),
        )
        all_metrics[split] = {
            "multi_view": metrics,
            "official_single_image": single_metrics,
        }
        logger.info(
            "%s single-image embedding to gallery images Recall@K: %s", split,
            single_metrics["image"]["joint"],
        )
        logger.info(
            "%s fused multi-image embedding to gallery images Recall@K: %s", split,
            metrics["image"]["joint"],
        )
        logger.info("%s retrieval metrics:\n%s", split,
                    json.dumps(all_metrics[split], indent=2, sort_keys=True))

    destination = Path(args.save_dir) / "retrieval_test_metrics.json"
    _write_json(destination, all_metrics)
    logger.info("Saved retrieval metrics to %s", destination)
    return all_metrics


def train(args, device, logger):
    train_dataset = OpenHotelsRetrievalTrainDataset(
        args.data_dir, split="gallery_train",
        min_query_views=args.min_query_views,
        max_query_views=args.max_query_views,
        positive_gallery_size=args.positive_gallery_size,
        val_fraction=args.val_fraction, seed=args.seed,
        skip_missing=args.skip_missing, image_size=args.image_size,
    )
    logger.info(
        "Retrieval training: %d eligible hotels, %d excluded hotels, "
        "%d unavailable images", len(train_dataset),
        train_dataset.excluded_hotel_count, train_dataset.missing_count,
    )
    batch_sampler = HotelBalancedBatchSampler(
        train_dataset, hotels_per_batch=args.hotels_per_batch,
        steps_per_epoch=args.steps_per_epoch or None, seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        collate_fn=retrieval_collate_fn, **_loader_kwargs(args),
    )

    model, checkpoint_payload = build_model(
        args, train_dataset.num_classes, device, logger,
        checkpoint=args.resume or args.checkpoint, return_checkpoint=True,
        classes=train_dataset.classes,
    )
    criterion = EasyPositiveLoss(
        mode=args.loss, temperature=args.temperature,
        semi_hard_fallback=args.semi_hard_fallback,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    trainer = RetrievalTrainer(
        model=model, criterion=criterion, optimizer=optimizer, device=device,
        logger=logger, save_dir=args.save_dir,
        view_loss_weight=args.view_loss_weight,
        classification_loss_weight=args.classification_loss_weight,
        classification_decay_epochs=args.classification_decay_epochs,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip_norm=args.grad_clip_norm,
        log_interval=args.log_interval,
        checkpoint_metadata={
            "classes": list(train_dataset.classes),
            "config": {
                "dataset": "openhotels",
                "architecture": args.architecture,
                "image_size": args.image_size,
                "model_max_views": args.model_max_views,
            },
        },
    )
    start_epoch, best_score = 1, float("-inf")
    if args.resume:
        start_epoch, best_score = trainer.restore_training_state(checkpoint_payload)
    completed_epochs = start_epoch - 1
    final_epoch = (
        completed_epochs + args.additional_epochs
        if args.additional_epochs else args.num_epochs
    )
    optimizer_steps = math.ceil(len(train_loader) / args.grad_accum_steps)
    trainer.scheduler = build_retrieval_scheduler(
        optimizer, max_lr=args.lr, epochs=args.num_epochs,
        steps_per_epoch=optimizer_steps, completed_epochs=completed_epochs,
        additional_epochs=args.additional_epochs,
        scheduler_phase=trainer.scheduler_phase,
    )
    if args.additional_epochs:
        logger.info(
            "Starting a new %d-epoch OneCycle phase from retrieval checkpoint "
            "epoch %d; global epochs %d-%d, phase scheduler step %d/%d, "
            "%d optimizer steps per epoch, lr=%.6g, best_score=%s",
            args.additional_epochs, completed_epochs, start_epoch, final_epoch,
            trainer.scheduler.last_epoch, trainer.scheduler.total_steps,
            optimizer_steps, optimizer.param_groups[0]["lr"], best_score,
        )
    elif args.resume:
        logger.info(
            "Resuming retrieval at epoch %d/%d with %d optimizer steps per epoch, "
            "lr=%.6g, best_score=%s", start_epoch, args.num_epochs,
            optimizer_steps, optimizer.param_groups[0]["lr"], best_score,
        )
        if trainer.scheduler_phase is not None:
            logger.info(
                "Resuming existing OneCycle phase at phase scheduler step %d/%d "
                "with global phase start epoch %d",
                trainer.scheduler.last_epoch, trainer.scheduler.total_steps,
                trainer.scheduler_phase["start_epoch"],
            )

    validation_evaluator = RetrievalEvaluator(
        model=model, device=device, logger=logger,
        log_interval=args.eval_log_interval,
    )
    if args.skip_validation:
        validate_fn = None
    else:
        val_gallery, val_queries = build_openhotels_seen_validation(
            args.data_dir, classes=train_dataset.classes,
            val_fraction=args.val_fraction,
            min_views=args.min_query_views,
            max_views=args.max_query_views,
            seed=args.seed, skip_missing=args.skip_missing,
            image_size=args.image_size,
        )
        val_gallery = _cap_gallery_per_hotel(
            val_gallery, args.val_gallery_images_per_hotel,
        )
        val_queries = _subset_queries(
            val_queries, args.max_val_queries, args.seed + 17,
        )
        val_gallery_loader, val_query_loader = _make_eval_loaders(
            val_gallery, val_queries, args,
        )
        logger.info(
            "Validation retrieval: %d gallery images, %d upload queries",
            len(val_gallery), len(val_queries),
        )

        def validate_fn(current_model, epoch):
            validation_evaluator.model = current_model
            gallery = validation_evaluator.extract_gallery(val_gallery_loader)
            queries = validation_evaluator.extract_queries(val_query_loader)
            metrics = validation_evaluator.evaluate_embeddings(
                queries["joint_embeddings"], queries["view_embeddings"],
                queries["labels"], gallery["embeddings"], gallery["labels"],
                query_view_mask=queries["view_mask"],
                **_evaluation_kwargs(args, compute_image_metrics=True),
            )
            _write_json(
                Path(args.save_dir) / f"validation-epoch-{epoch:03d}.json",
                metrics,
            )
            return metrics, _best_validation_score(metrics)

    best_path = trainer.fit(
        train_loader, final_epoch, validate_fn=validate_fn,
        eval_every=args.eval_every, start_epoch=start_epoch, best_score=best_score,
    )
    _load_checkpoint(model, best_path, logger)
    return model, train_dataset.classes, best_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-view OpenHotels retrieval training",
        allow_abbrev=False,
    )
    parser.add_argument("--data_dir", default="data/OpenHotels-Updated")
    parser.add_argument("--save_dir", default="output/openhotels-retrieval")
    parser.add_argument(
        "--checkpoint",
        default="output/openhotels-classification/best.pth",
        help="classification or retrieval checkpoint used to initialize the model",
    )
    parser.add_argument(
        "--resume", default="",
        help=("retrieval checkpoint whose model, optimizer, completed epochs and "
              "best score are restored; overrides --checkpoint"),
    )
    parser.add_argument(
        "--additional_epochs", type=int, default=0,
        help=("with --resume, start a new OneCycle phase for this many additional "
              "epochs; global epoch numbers continue from the checkpoint and "
              "--num_epochs is ignored for the phase duration"),
    )
    parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--skip_test", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--architecture", default="vit_small_r26_s32_224")
    parser.add_argument("--pretrained_weights", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--model_max_views", type=int, default=4,
        help=("number of learned image-position slots in the model; keep this at "
              "4 when loading the supplied four-view classification checkpoint"),
    )
    parser.add_argument("--min_query_views", type=int, default=2)
    parser.add_argument("--max_query_views", type=int, default=4)
    parser.add_argument("--positive_gallery_size", type=int, default=4)
    parser.add_argument("--hotels_per_batch", type=int, default=16)
    parser.add_argument("--steps_per_epoch", type=int, default=0,
                        help="zero visits every eligible hotel in a shuffled pass")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--loss", choices=("ep", "ephn", "epshn"),
                        default="epshn")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--semi_hard_fallback", choices=("hardest", "skip"),
                        default="hardest")
    parser.add_argument("--view_loss_weight", type=float, default=0.25)
    parser.add_argument("--classification_loss_weight", type=float, default=0.1)
    parser.add_argument("--classification_decay_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_clip_norm", type=float, default=80.0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--gallery_batch_size", type=int, default=128)
    parser.add_argument("--query_batch_size", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_log_interval", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument(
        "--skip_validation", "--no_validation", action="store_true",
        help="train without the upload-disjoint seen-hotel validation pass",
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_val_queries", type=int, default=2000)
    parser.add_argument("--val_gallery_images_per_hotel", type=int, default=4,
                        help="zero keeps the complete upload-disjoint validation gallery")
    parser.add_argument("--max_test_queries", type=int, default=0,
                        help="zero evaluates every official query group")
    parser.add_argument("--test_min_views", type=int, default=2,
                        help="minimum views in the primary multi-view test")
    parser.add_argument("--max_test_image_queries", type=int, default=0,
                        help="zero evaluates every official single-image query")
    parser.add_argument("--test_collections_per_group", type=int, default=1)
    parser.add_argument("--recall_at", type=int, nargs="+", default=(1, 5, 10, 100))
    parser.add_argument("--hotel_top_m", type=int, nargs="+", default=(1, 3))
    parser.add_argument("--hybrid_alpha", type=float, default=0.5)
    parser.add_argument("--retrieval_query_chunk_size", type=int, default=16)
    parser.add_argument("--retrieval_gallery_chunk_size", type=int, default=32768)
    parser.add_argument("--embedding_cache_dir", default="")
    parser.add_argument("--reuse_embedding_cache",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_missing", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or a device such as cuda:0")
    return parser.parse_args()


def validate_args(args):
    if args.resume and args.eval_only:
        raise ValueError("--resume is for training; use --checkpoint with --eval_only")
    if args.additional_epochs < 0:
        raise ValueError("additional_epochs cannot be negative")
    if args.additional_epochs and not args.resume:
        raise ValueError("--additional_epochs requires --resume")
    if args.model_max_views < 1:
        raise ValueError("model_max_views must be positive")
    if not 1 <= args.min_query_views <= args.max_query_views:
        raise ValueError("require 1 <= min_query_views <= max_query_views")
    if args.max_query_views > args.model_max_views:
        raise ValueError("max_query_views cannot exceed model_max_views")
    if not 1 <= args.test_min_views <= args.max_query_views:
        raise ValueError("require 1 <= test_min_views <= max_query_views")
    if args.positive_gallery_size < 1:
        raise ValueError("positive_gallery_size must be at least 1")
    if args.hotels_per_batch < 2:
        raise ValueError("hotels_per_batch must be at least 2 for metric mining")
    if args.num_epochs < 1 or args.grad_accum_steps < 1:
        raise ValueError("num_epochs and grad_accum_steps must be positive")
    if args.steps_per_epoch < 0:
        raise ValueError("steps_per_epoch cannot be negative")
    if args.eval_every < 1 or args.log_interval < 1 or args.eval_log_interval < 1:
        raise ValueError("evaluation and logging intervals must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("num_workers must be non-negative and prefetch_factor positive")
    if args.gallery_batch_size < 1 or args.query_batch_size < 1:
        raise ValueError("evaluation batch sizes must be positive")
    if args.retrieval_query_chunk_size < 1 or args.retrieval_gallery_chunk_size < 1:
        raise ValueError("retrieval chunk sizes must be positive")
    if not 0 <= args.hybrid_alpha <= 1:
        raise ValueError("hybrid_alpha must lie in [0, 1]")
    if not 0 <= args.val_fraction < 1:
        raise ValueError("val_fraction must lie in [0, 1)")
    for name in (
        "max_val_queries",
        "val_gallery_images_per_hotel",
        "max_test_queries",
        "max_test_image_queries",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    if args.test_collections_per_group < 1:
        raise ValueError("test_collections_per_group must be positive")
    if any(value < 1 for value in tuple(args.recall_at) + tuple(args.hotel_top_m)):
        raise ValueError("recall_at and hotel_top_m values must be positive")
    if args.view_loss_weight < 0 or args.classification_loss_weight < 0:
        raise ValueError("loss weights cannot be negative")
    if args.classification_decay_epochs < 0:
        raise ValueError("classification_decay_epochs cannot be negative")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive")


def main():
    args = parse_args()
    validate_args(args)
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = str(save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(save_dir / "retrieval.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("openhotels-retrieval")
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    logger.info("Using device %s", device)
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.eval_only:
        gallery_dataset = OpenHotelsGalleryDataset(
            args.data_dir, split="gallery", seed=args.seed,
            skip_missing=args.skip_missing, image_size=args.image_size,
        )
        classes = gallery_dataset.classes
        model = build_model(
            args, gallery_dataset.num_classes, device, logger,
            checkpoint=args.checkpoint,
            classes=classes,
        )
        checkpoint_path = args.checkpoint
    else:
        model, classes, checkpoint_path = train(args, device, logger)

    if not args.skip_test:
        checkpoint_tag = (
            _embedding_cache_tag(args, checkpoint_path)
            if args.embedding_cache_dir else "uncached"
        )
        run_official_evaluation(
            model, classes, args, device, logger,
            checkpoint_tag=checkpoint_tag,
        )


if __name__ == "__main__":
    main()
