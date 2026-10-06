"""Small end-to-end OpenHotels data/model/loss smoke test."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import OpenHotelsDataset  # noqa: E402
from loss import MutualDistillationLoss  # noqa: E402
from model import MultiImageHybrid  # noqa: E402


def main(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset = OpenHotelsDataset(
        args.data_dir, split="gallery_train", n=args.num_images,
        train=True, image_size=args.image_size, skip_missing=True,
    )
    images, _, paths = dataset[0]
    expected = (args.num_images, 3, args.image_size, args.image_size)
    if tuple(images.shape) != expected:
        raise AssertionError(f"Expected image shape {expected}, got {tuple(images.shape)}")

    images = images.unsqueeze(0).repeat(args.batch_size, 1, 1, 1, 1).to(device)
    targets = torch.zeros((args.batch_size, args.num_images), dtype=torch.long,
                          device=device)
    model = MultiImageHybrid(
        args.architecture, num_classes=args.smoke_classes, n=args.num_images,
        pretrained_weights=False, image_size=args.image_size,
    ).to(device)
    outputs = model(images)
    expected_single = (args.batch_size * args.num_images, args.smoke_classes)
    expected_collection = (args.batch_size, args.smoke_classes)
    if tuple(outputs["single"]["logits"].shape) != expected_single:
        raise AssertionError("Unexpected single-view output shape")
    if tuple(outputs["mv_collection"]["logits"].shape) != expected_collection:
        raise AssertionError("Unexpected collection output shape")

    criterion = nn.CrossEntropyLoss()
    single_logits = outputs["single"]["logits"]
    collection_logits = outputs["mv_collection"]["logits"]
    base_loss = criterion(single_logits, targets.flatten())
    base_loss += criterion(collection_logits, targets[:, 0])
    md_loss = MutualDistillationLoss(temp=4.0, lambda_hyperparam=0.1)(
        collection_logits,
        single_logits.reshape(args.batch_size, args.num_images, args.smoke_classes),
        targets[:, 0],
    )
    loss = base_loss + md_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.smoke_classes == dataset.num_classes:
        production_head_shape = tuple(model.model.head.weight.shape)
    else:
        production_model = MultiImageHybrid(
            args.architecture, num_classes=dataset.num_classes, n=args.num_images,
            pretrained_weights=False, image_size=args.image_size,
        )
        production_head_shape = tuple(production_model.model.head.weight.shape)
    print(f"PyTorch: {torch.__version__} (CUDA build {torch.version.cuda})")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Peak allocated: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    print(f"Available training samples: {len(dataset):,}")
    print(f"Available hotel classes: {len(dataset.targets):,}/{dataset.num_classes:,}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Example paths: {paths}")
    print(f"Single/collection logits: {expected_single} / {expected_collection}")
    print(f"Production head: {production_head_shape}")
    print(f"Base/MD/total loss: {base_loss.item():.6f} / {md_loss.item():.6f} / {loss.item():.6f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "OpenHotels-Updated"))
    parser.add_argument("--architecture", default="vit_small_r26_s32_224")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--smoke_classes", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    main(parser.parse_args())
