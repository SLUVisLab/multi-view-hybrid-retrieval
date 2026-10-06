"""Train and independently evaluate the MV-HFMD hotel classifier."""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from datasets import HotelsDataset, OpenHotelsDataset
from engine import Evaluator, TrainerEngine
from loss import MutualDistillationLoss
from model import MultiImageHybrid
from utils import str2bool


def _read_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location='cpu')


def _model_config(args):
    return {
        'dataset': args.dataset,
        'architecture': args.architecture,
        'image_size': args.image_size,
        'model_max_views': args.num_images,
    }


def _validate_checkpoint_metadata(checkpoint, classes, args):
    saved_classes = checkpoint.get('classes')
    if saved_classes is not None and list(map(str, saved_classes)) != list(map(str, classes)):
        raise ValueError('Checkpoint hotel classes/order do not match the gallery metadata')
    saved_config = checkpoint.get('config', {})
    for key, value in _model_config(args).items():
        if key in saved_config and saved_config[key] != value:
            raise ValueError(
                f'Checkpoint {key}={saved_config[key]!r} does not match requested {value!r}')


def _load_model_weights(model, checkpoint):
    state = checkpoint.get('model', checkpoint)
    model.load_state_dict({
        key.removeprefix('module.'): value for key, value in state.items()
    }, strict=True)


def _evaluation_classes(args):
    """Reconstruct legacy class order without a training dataset or loader."""
    if args.dataset == 'openhotels':
        with (Path(args.data_dir) / 'metadata_gallery.json').open() as handle:
            rows = json.load(handle)
        return sorted({row['hotel_id'] for row in rows})
    paths = np.load(Path(args.data_dir) / 'train.npy').tolist()
    return sorted({str(path).split('/')[-2] for path in paths})


def _loader_kwargs(args, device):
    kwargs = dict(
        num_workers=args.num_workers, drop_last=False,
        pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        kwargs['prefetch_factor'] = args.prefetch_factor
    return kwargs


def _build_scheduler(optimizer, args, num_batches):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=args.lr, epochs=args.num_epochs,
        div_factor=10, steps_per_epoch=math.ceil(num_batches / args.grad_accum_steps),
        final_div_factor=1000, pct_start=min(0.3, 5 / args.num_epochs),
        anneal_strategy='cos',
    )


@torch.no_grad()
def evaluate_test_splits(model, classes, args, device, logger):
    evaluator = Evaluator(model=model, n=args.num_images, device=device,
                          logger=logger, log_interval=args.log_interval)
    test_splits = ['test_room', 'test_object'] if args.dataset == 'openhotels' else ['test']
    all_scores = {}
    for test_split in test_splits:
        try:
            dataset_test = (OpenHotelsDataset(
                args.data_dir, split=test_split, n=args.num_images, classes=classes,
                max_collections_per_class=args.max_test_collections_per_class,
                skip_missing=args.skip_missing, image_size=args.image_size)
                if args.dataset == 'openhotels' else
                HotelsDataset(args.data_dir, split=test_split, n=args.num_images,
                              classes=classes, train=False, image_size=args.image_size))
        except RuntimeError as error:
            logger.warning('Skipping %s evaluation: %s', test_split, error)
            continue
        loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.eval_batch_size, shuffle=False,
            **_loader_kwargs(args, device))
        scores = evaluator.evaluate(loader_test)
        all_scores[test_split] = scores
        for view_type, metrics in scores.items():
            for metric, value in metrics.items():
                logger.info('%s %s %s: %s', test_split, view_type, metric, value)
    destination = Path(args.save_dir) / 'classification_test_metrics.json'
    with destination.open('w') as handle:
        json.dump(all_scores, handle, indent=2, sort_keys=True)
    logger.info('Saved classification metrics to %s', destination)
    return all_scores


def main(args=None, logger=None):
    args = args or parse_args()
    logger = logger or logging.getLogger('classification')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = _read_checkpoint(args.checkpoint) if args.checkpoint else None

    if args.eval_only:
        classes = _evaluation_classes(args)
        _validate_checkpoint_metadata(checkpoint, classes, args)
        model = MultiImageHybrid(
            args.architecture, num_classes=len(classes), n=args.num_images,
            pretrained_weights=False, image_size=args.image_size)
        _load_model_weights(model, checkpoint)
        model.to(device)
        return evaluate_test_splits(model, classes, args, device, logger)

    if args.dataset == 'openhotels':
        dataset_train = OpenHotelsDataset(
            args.data_dir, split='gallery_train', n=args.num_images, train=True,
            val_fraction=args.val_fraction, seed=args.seed,
            skip_missing=args.skip_missing, image_size=args.image_size)
        dataset_val = OpenHotelsDataset(
            args.data_dir, split='gallery_val', n=args.num_images,
            classes=dataset_train.classes, val_fraction=args.val_fraction,
            seed=args.seed, max_collections_per_class=args.max_val_collections_per_class,
            skip_missing=args.skip_missing, image_size=args.image_size)
        logger.info('OpenHotels gallery: %d training samples, %d validation collections, '
                    '%d unavailable training rows, %d unavailable validation rows',
                    len(dataset_train), len(dataset_val), dataset_train.missing_count,
                    dataset_val.missing_count)
    else:
        dataset_train = HotelsDataset(args.data_dir, split='train', n=args.num_images,
                                      train=True, image_size=args.image_size)
        dataset_val = HotelsDataset(args.data_dir, split='val', n=args.num_images,
                                    classes=dataset_train.classes, train=False,
                                    image_size=args.image_size)
    classes = dataset_train.classes
    if checkpoint is not None:
        _validate_checkpoint_metadata(checkpoint, classes, args)
    model = MultiImageHybrid(
        args.architecture, num_classes=len(classes), n=args.num_images,
        pretrained_weights=args.pretrained_weights and checkpoint is None,
        image_size=args.image_size)
    if checkpoint is not None:
        _load_model_weights(model, checkpoint)
    model.to(device)
    logger.info('Number of classes: %d', len(classes))

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        **_loader_kwargs(args, device))
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.eval_batch_size, shuffle=False,
        **_loader_kwargs(args, device))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd,
                                momentum=args.momentum)
    scheduler = _build_scheduler(optimizer, args, len(loader_train))
    md_loss = (MutualDistillationLoss(temp=args.md_temp, lambda_hyperparam=args.md_lambda)
               if args.use_mutual_distillation_loss else None)
    evaluator = Evaluator(model=model, n=args.num_images, device=device,
                          logger=logger, log_interval=args.log_interval)
    trainer = TrainerEngine(
        model=model, lr_scheduler_model=scheduler, criterion=nn.CrossEntropyLoss(ignore_index=-1),
        optimizer_model=optimizer, evaluator=evaluator, md_loss=md_loss,
        grad_clip_norm=args.grad_clip_norm, logger=logger, save_dir=args.save_dir,
        grad_accum_steps=args.grad_accum_steps, log_interval=args.log_interval,
        eval_every=args.eval_every, skip_initial_eval=args.skip_initial_eval)
    trainer.metadata.update({'classes': list(classes), 'config': _model_config(args)})
    trainer.train(loader_train, args.num_epochs, loader_val)
    logger.info('Training done!')
    _load_model_weights(model, _read_checkpoint(Path(args.save_dir) / 'best.pth'))
    return evaluate_test_splits(model, classes, args, device, logger)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Multi-view classification training and evaluation', allow_abbrev=False)
    parser.add_argument('--data_dir', default='data/OpenHotels-Updated', help='dataset directory')
    parser.add_argument('--dataset', choices=['hotels8k', 'openhotels'], default='openhotels')
    parser.add_argument('--architecture', default='vit_small_r26_s32_224')
    parser.add_argument('--pretrained_weights', default=True, type=str2bool)
    parser.add_argument('--save_dir', default='output/classification')
    parser.add_argument('--checkpoint', default='', help='checkpoint for evaluation or model initialization')
    parser.add_argument('--eval_only', action='store_true', help='evaluate --checkpoint without training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--num_images', default=4, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--prefetch_factor', default=2, type=int)
    parser.add_argument('--grad_accum_steps', default=1, type=int)
    parser.add_argument('--eval_every', default=5, type=int)
    parser.add_argument('--skip_initial_eval', default=True, type=str2bool)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--val_fraction', default=0.1, type=float)
    parser.add_argument('--max_val_collections_per_class', default=1, type=int,
                        help='cap validation collections per hotel; -1 removes the cap')
    parser.add_argument('--max_test_collections_per_class', default=None, type=int)
    parser.add_argument('--skip_missing', default=True, type=str2bool)
    parser.add_argument('--use_mutual_distillation_loss', default=True, type=str2bool)
    parser.add_argument('--md_temp', default=4., type=float)
    parser.add_argument('--md_lambda', default=.1, type=float)
    parser.add_argument('--grad_clip_norm', default=80., type=float)
    args = parser.parse_args(argv)
    if args.eval_only and not args.checkpoint:
        parser.error('--eval_only requires --checkpoint')
    for name in ('batch_size', 'eval_batch_size', 'num_epochs', 'num_images', 'image_size',
                 'prefetch_factor', 'grad_accum_steps', 'eval_every', 'log_interval'):
        if getattr(args, name) < 1:
            parser.error(f'--{name} must be positive')
    if args.num_workers < 0:
        parser.error('--num_workers must be non-negative')
    if not 0 <= args.val_fraction < 1:
        parser.error('--val_fraction must be in [0, 1)')
    if args.max_val_collections_per_class is not None and args.max_val_collections_per_class < 0:
        args.max_val_collections_per_class = None
    return args


if __name__ == '__main__':
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(Path(args.save_dir) / 'log.txt'), logging.StreamHandler()])
    logger = logging.getLogger('classification')
    for key, value in vars(args).items():
        logger.info('%s: %s', key, value)
    main(args, logger)
