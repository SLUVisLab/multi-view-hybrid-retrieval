"""One-step retrieval smoke test using real OpenHotels images.

The optimization step deliberately uses a tiny 32-pixel model so it can run on
CPU in seconds.  By default the script also strictly loads the production
four-view checkpoint and performs one 224-pixel gallery embedding forward.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import (  # noqa: E402
    HotelBalancedBatchSampler,
    OpenHotelsGalleryDataset,
    OpenHotelsRetrievalTrainDataset,
    retrieval_collate_fn,
)
from loss import EasyPositiveLoss  # noqa: E402
from model import MultiImageHybrid  # noqa: E402


def _load_model_checkpoint(model, checkpoint):
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del payload, state


def _embedding(outputs, branch):
    return outputs[branch]["embeddings"]


def run_tiny_training_step(args, device):
    dataset = OpenHotelsRetrievalTrainDataset(
        args.data_dir,
        split="gallery_train",
        min_query_views=2,
        max_query_views=4,
        positive_gallery_size=4,
        val_fraction=0.1,
        seed=args.seed,
        skip_missing=args.skip_missing,
        image_size=32,
    )
    sampler = HotelBalancedBatchSampler(
        dataset, hotels_per_batch=2, steps_per_epoch=1, seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=retrieval_collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))

    if batch["query_label"].unique().numel() != 2:
        raise AssertionError("The balanced batch did not contain distinct hotels")
    for query_paths, gallery_paths in zip(
        batch["query_paths"], batch["gallery_paths"]
    ):
        if set(query_paths) & set(gallery_paths):
            raise AssertionError("A query source path leaked into its positive gallery")

    model = MultiImageHybrid(
        "vit_tiny_patch16_224",
        num_classes=8,
        n=4,
        pretrained_weights=False,
        image_size=32,
    ).to(device)
    criterion = EasyPositiveLoss(mode="epshn", temperature=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    query_images = batch["query_images"].to(device)
    query_mask = batch["query_mask"].to(device)
    query_labels = batch["query_label"].to(device)
    gallery_images = batch["gallery_images"].to(device)
    gallery_mask = batch["gallery_mask"].to(device)
    gallery_labels = batch["gallery_labels"].to(device)[gallery_mask]

    query_output = model.forward_embeddings(query_images, query_mask)
    gallery_output = model.forward_embeddings(
        gallery_images[gallery_mask].unsqueeze(1), return_collection=False,
    )
    joint = _embedding(query_output, "mv_collection")
    views = _embedding(query_output, "single").reshape(
        query_images.shape[0], query_images.shape[1], -1
    )
    gallery = _embedding(gallery_output, "single")
    view_labels = query_labels[:, None].expand_as(query_mask)[query_mask]

    joint_loss = criterion(joint, gallery, query_labels, gallery_labels)
    view_loss = criterion(views[query_mask], gallery, view_labels, gallery_labels)
    loss = joint_loss + 0.25 * view_loss
    if not torch.isfinite(loss):
        raise AssertionError(f"Retrieval loss is not finite: {loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "dataset": dataset,
        "query_shape": tuple(query_images.shape),
        "query_counts": query_mask.sum(1).tolist(),
        "gallery_shape": tuple(gallery_images.shape),
        "gallery_counts": gallery_mask.sum(1).tolist(),
        "joint_loss": joint_loss.item(),
        "view_loss": view_loss.item(),
        "total_loss": loss.item(),
    }


@torch.inference_mode()
def check_production_checkpoint(args, dataset, device):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Production checkpoint not found: {checkpoint}")
    model = MultiImageHybrid(
        args.architecture,
        num_classes=dataset.num_classes,
        n=4,
        pretrained_weights=False,
        image_size=224,
    )
    _load_model_checkpoint(model, checkpoint)
    model = model.to(device).eval()

    gallery_dataset = OpenHotelsGalleryDataset(
        args.data_dir,
        split="gallery",
        classes=dataset.classes,
        skip_missing=args.skip_missing,
        image_size=224,
    )
    image = gallery_dataset[0]["image"].unsqueeze(0).unsqueeze(0).to(device)
    output = model.forward_embeddings(image, return_collection=False)
    embedding = output["single"]["embeddings"]
    expected = (1, model.embed_dim)
    if tuple(embedding.shape) != expected:
        raise AssertionError(
            f"Expected production embedding shape {expected}, got {tuple(embedding.shape)}"
        )
    if not torch.isfinite(embedding).all():
        raise AssertionError("Production embedding contains non-finite values")
    return tuple(embedding.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "OpenHotels-Updated"))
    parser.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT / "output" / "openhotels-classification" / "best.pth"
        ),
    )
    parser.add_argument("--architecture", default="vit_small_r26_s32_224")
    parser.add_argument(
        "--check_checkpoint", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--skip_missing", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    result = run_tiny_training_step(args, device)
    checkpoint_shape = None
    if args.check_checkpoint:
        checkpoint_shape = check_production_checkpoint(
            args, result["dataset"], device,
        )

    dataset = result.pop("dataset")
    print(f"Device: {device}")
    print(
        f"Eligible/excluded hotels: {len(dataset):,}/"
        f"{dataset.excluded_hotel_count:,}"
    )
    print(
        f"Query batch: {result['query_shape']} valid={result['query_counts']}"
    )
    print(
        f"Positive gallery batch: {result['gallery_shape']} "
        f"valid={result['gallery_counts']}"
    )
    print(
        "EPSHN losses: "
        f"joint={result['joint_loss']:.6f} view={result['view_loss']:.6f} "
        f"total={result['total_loss']:.6f}"
    )
    if checkpoint_shape is not None:
        print(f"Production checkpoint embedding: {checkpoint_shape}")
    print("RETRIEVAL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
