"""OpenHotels-Updated dataset support.

Images can be read from the release tar files or from the directory layout produced
by extracting each shard into ``images/<shard-stem>/``.  The latter is useful while
the dataset is still being transferred/extracted.
"""

import hashlib
import io
import json
import math
import os
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


def _stable_fraction(path, seed):
    value = hashlib.sha1(f"{seed}:{path}".encode("utf-8")).digest()[:8]
    return int.from_bytes(value, "big") / float(2**64)


class OpenHotelsDataset(torch.utils.data.Dataset):
    """Create multi-image hotel collections from OpenHotels metadata.

    ``gallery_train`` and ``gallery_val`` are a deterministic, per-hotel split of
    the gallery. Query splits use all rows whose hotel is represented by ``classes``.
    Missing shards are skipped by default, which makes the loader usable while an
    rsync is in progress. Recreate the dataset at the start of each run to discover
    newly arrived shards.
    """

    METADATA_FILES = {
        "gallery": "metadata_gallery.json",
        "gallery_train": "metadata_gallery.json",
        "gallery_val": "metadata_gallery.json",
        "test_room": "metadata_test_room.json",
        "test_object": "metadata_test_object.json",
    }

    def __init__(self, data_dir, split, n=2, train=False, classes=None,
                 val_fraction=0.1, seed=0, max_collections_per_class=None,
                 skip_missing=True, image_size=224):
        if split not in self.METADATA_FILES:
            raise ValueError(f"Unknown OpenHotels split: {split}")
        if n < 1:
            raise ValueError("n must be at least 1")

        self.data_dir = Path(data_dir)
        self.split = split
        self.n = n
        self.train = train
        self.seed = seed
        self.skip_missing = skip_missing
        self.max_collections_per_class = max_collections_per_class
        self.image_size = image_size
        self._tar_handles = {}

        metadata_path = self.data_dir / self.METADATA_FILES[split]
        with metadata_path.open("r") as handle:
            rows = json.load(handle)
        if split.startswith("gallery_"):
            rows = self._partition_gallery(rows, split, val_fraction, seed)

        if classes is None:
            self.classes = sorted({row["hotel_id"] for row in rows})
        else:
            self.classes = list(classes)
        self.class_to_idx = {hotel_id: i for i, hotel_id in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Walk each extracted shard once instead of issuing hundreds of thousands
        # of random stat calls on a shared filesystem. This also distinguishes a
        # partially rsynced shard directory from a complete one.
        available_members = self._index_available_members(rows) if skip_missing else None
        grouped = defaultdict(list)
        missing = 0
        for row in rows:
            target = self.class_to_idx.get(row["hotel_id"])
            if target is None:
                continue
            if skip_missing:
                members = available_members.get(row["shard"], False)
                if members is False or (members is not None and row["path"] not in members):
                    missing += 1
                    continue
            grouped[target].append(row)
        self.missing_count = missing
        self.rows_by_target = dict(grouped)
        self.targets = sorted(self.rows_by_target)
        if not self.targets:
            raise RuntimeError(
                f"No available images found for {split} under {self.data_dir}. "
                "The transfer may not have reached the required shards yet."
            )

        if train:
            # One sample per source image preserves natural class rebalancing from
            # the original code; companion views are sampled from the same hotel.
            self.samples = [(target, index)
                            for target in self.targets
                            for index in range(len(self.rows_by_target[target]))]
        else:
            self.samples = self._make_eval_collections()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        resize_size = round(image_size * 256 / 224)
        if train:
            self.transform = transforms.Compose([
                transforms.Resize(resize_size), transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resize_size), transforms.CenterCrop(image_size),
                transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
            ])

    @staticmethod
    def _partition_gallery(rows, split, val_fraction, seed):
        if not 0 <= val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        by_hotel = defaultdict(list)
        for row in rows:
            by_hotel[row["hotel_id"]].append(row)
        selected = []
        for hotel_rows in by_hotel.values():
            ordered = sorted(hotel_rows, key=lambda r: _stable_fraction(r["path"], seed))
            n_val = min(len(ordered) - 1, int(math.ceil(len(ordered) * val_fraction)))
            n_val = max(0, n_val)
            selected.extend(ordered[-n_val:] if split == "gallery_val" and n_val else
                            ordered[:-n_val] if split == "gallery_train" and n_val else
                            [] if split == "gallery_val" else ordered)
        return selected

    def _extracted_path(self, row):
        shard_stem = Path(row["shard"]).stem
        return self.data_dir / "images" / shard_stem / row["path"]

    def _index_available_members(self, rows):
        """Map shards to extracted member sets; ``None`` means a tar is available."""
        result = {}
        for shard in {row["shard"] for row in rows}:
            tar_path = self.data_dir / shard
            if tar_path.is_file():
                result[shard] = None
                continue
            shard_dir = self.data_dir / "images" / Path(shard).stem
            if not shard_dir.is_dir():
                continue
            members = set()
            for directory, _, filenames in os.walk(shard_dir):
                relative_dir = Path(directory).relative_to(shard_dir)
                members.update(str(relative_dir / filename) for filename in filenames)
            result[shard] = members
        return result

    def _make_eval_collections(self):
        samples = []
        for target in self.targets:
            count = len(self.rows_by_target[target])
            num_collections = int(math.ceil(count / self.n))
            if self.max_collections_per_class is not None:
                num_collections = min(num_collections, self.max_collections_per_class)
            for collection_index in range(num_collections):
                indices = [(collection_index * self.n + j) % count for j in range(self.n)]
                samples.append((target, indices))
        return samples

    def _load_image(self, row):
        extracted = self._extracted_path(row)
        if extracted.is_file():
            with Image.open(extracted) as image:
                return image.convert("RGB")

        shard_path = self.data_dir / row["shard"]
        key = str(shard_path)
        tar = self._tar_handles.get(key)
        if tar is None:
            tar = tarfile.open(shard_path, "r")
            self._tar_handles[key] = tar
        member = tar.extractfile(row["path"])
        if member is None:
            raise FileNotFoundError(f"{row['path']} not found in {shard_path}")
        with Image.open(io.BytesIO(member.read())) as image:
            return image.convert("RGB")

    def __getitem__(self, index):
        target, item = self.samples[index]
        rows = self.rows_by_target[target]
        if self.train:
            rng = np.random.default_rng(self.seed + index + os.getpid())
            choices = [item]
            if self.n > 1:
                choices.extend(rng.choice(len(rows), self.n - 1,
                                          replace=len(rows) < self.n).tolist())
        else:
            choices = item
        selected = [rows[i] for i in choices]
        images = torch.stack([self.transform(self._load_image(row)) for row in selected])
        targets = torch.full((self.n,), target, dtype=torch.long)
        paths = [row["path"] for row in selected]
        return images, targets, paths

    def __len__(self):
        return len(self.samples)

    def __del__(self):
        for handle in getattr(self, "_tar_handles", {}).values():
            handle.close()
