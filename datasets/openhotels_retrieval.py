"""Retrieval-oriented OpenHotels datasets and balanced episode sampling.

The classification loader in :mod:`datasets.openhotels` treats every image as a
training sample and chooses companion images from the same hotel.  Retrieval
training needs stronger structure:

* a query is a coherent upload, identified by ``(hotel_id, room)``;
* query images and positive gallery images never contain the same source path;
* positives from another upload of the hotel are preferred;
* each metric-learning batch contains distinct, uniformly sampled hotels; and
* validation holds out complete uploads rather than leaking nearby images from
  an upload into both query and gallery.

All image sets are variable length.  ``retrieval_collate_fn`` pads them and
returns boolean masks, so images are never duplicated merely to fill four view
slots.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms


METADATA_FILES = {
    "gallery": "metadata_gallery.json",
    "gallery_train": "metadata_gallery.json",
    "gallery_val": "metadata_gallery.json",
    "test_room": "metadata_test_room.json",
    "test_object": "metadata_test_object.json",
}


def _stable_digest(value: str, seed: int) -> bytes:
    return hashlib.sha1(f"{seed}:{value}".encode("utf-8")).digest()


def _stable_order(value: str, seed: int) -> int:
    return int.from_bytes(_stable_digest(value, seed)[:8], "big")


def _room_key(row: Mapping) -> str:
    """Return an upload identifier, keeping missing room ids source-disjoint.

    Only two gallery images currently have an empty room id.  Treating every
    empty value as one upload could create a false multi-view query, so each
    missing-room image becomes its own group.
    """

    room = row.get("room")
    if room is None or str(room) == "":
        return f"__missing_room__:{row['path']}"
    return str(room)


def _group_key(row: Mapping) -> Tuple[str, str]:
    return str(row["hotel_id"]), _room_key(row)


def load_openhotels_metadata(data_dir: str | Path, split: str) -> List[dict]:
    """Load one OpenHotels metadata split.

    ``gallery_train`` and ``gallery_val`` both load gallery metadata; their
    upload-safe partition is applied by the dataset constructors below.
    """

    if split not in METADATA_FILES:
        raise ValueError(f"Unknown OpenHotels retrieval split: {split}")
    path = Path(data_dir) / METADATA_FILES[split]
    with path.open("r") as handle:
        return json.load(handle)


def partition_gallery_uploads(
    rows: Sequence[Mapping],
    val_fraction: float = 0.1,
    seed: int = 0,
    min_query_views: int = 2,
) -> Tuple[List[dict], List[dict]]:
    """Split gallery rows into reference and validation-query uploads.

    At least one upload remains in the reference gallery.  Only uploads with
    ``min_query_views`` images can be selected for validation, and hotels with
    fewer than two uploads are omitted from validation instead of leaking
    images from the same upload into both sides.
    """

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if min_query_views < 1:
        raise ValueError("min_query_views must be at least 1")

    groups_by_hotel: Dict[str, Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for original in rows:
        row = dict(original)
        groups_by_hotel[str(row["hotel_id"])][_room_key(row)].append(row)

    reference_rows: List[dict] = []
    query_rows: List[dict] = []
    for hotel_id in sorted(groups_by_hotel):
        groups = groups_by_hotel[hotel_id]
        eligible = [
            room for room, group_rows in groups.items()
            if len(group_rows) >= min_query_views
        ]
        if len(groups) < 2 or not eligible or val_fraction == 0:
            selected = set()
        else:
            desired = max(1, int(math.ceil(len(groups) * val_fraction)))
            desired = min(desired, len(groups) - 1, len(eligible))
            ordered = sorted(
                eligible,
                key=lambda room: _stable_order(f"{hotel_id}:{room}", seed),
            )
            selected = set(ordered[-desired:])

        for room, group_rows in groups.items():
            destination = query_rows if room in selected else reference_rows
            destination.extend(group_rows)

    return reference_rows, query_rows


def build_retrieval_transform(train: bool, image_size: int = 224):
    """Build the ImageNet-normalized transform used by the base model."""

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize_size = round(image_size * 256 / 224)
    if train:
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class _OpenHotelsImageStore:
    """Image loading shared by retrieval datasets.

    The preferred layout is ``images/<shard-stem>/<metadata path>`` produced by
    extracting each release shard.  If an extracted image is absent, loading
    falls back to the original tar shard.  Tar handles are local to each data
    loader worker.
    """

    def _init_store(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._tar_handles: Dict[str, tarfile.TarFile] = {}

    def _extracted_path(self, row: Mapping) -> Path:
        return (
            self.data_dir
            / "images"
            / Path(row["shard"]).stem
            / row["path"]
        )

    def _load_image(self, row: Mapping) -> Image.Image:
        extracted = self._extracted_path(row)
        if extracted.is_file():
            with Image.open(extracted) as image:
                return image.convert("RGB")

        shard_path = self.data_dir / row["shard"]
        key = str(shard_path)
        handle = self._tar_handles.get(key)
        if handle is None:
            handle = tarfile.open(shard_path, "r")
            self._tar_handles[key] = handle
        member = handle.extractfile(row["path"])
        if member is None:
            raise FileNotFoundError(f"{row['path']} not found in {shard_path}")
        with Image.open(io.BytesIO(member.read())) as image:
            return image.convert("RGB")

    def _available_rows(
        self, rows: Sequence[Mapping], skip_missing: bool
    ) -> Tuple[List[dict], int]:
        if not skip_missing:
            return [dict(row) for row in rows], 0

        # Index each extracted shard once.  Hundreds of thousands of individual
        # stat calls are particularly expensive on the shared filesystem.
        availability: Dict[str, Optional[set]] = {}
        for shard in {str(row["shard"]) for row in rows}:
            tar_path = self.data_dir / shard
            if tar_path.is_file():
                availability[shard] = None
                continue
            shard_dir = self.data_dir / "images" / Path(shard).stem
            if not shard_dir.is_dir():
                continue
            members = set()
            for directory, _, filenames in os.walk(shard_dir):
                relative = Path(directory).relative_to(shard_dir)
                members.update(str(relative / name) for name in filenames)
            availability[shard] = members

        selected: List[dict] = []
        missing = 0
        for original in rows:
            row = dict(original)
            members = availability.get(str(row["shard"]), False)
            if members is False or (
                members is not None and str(row["path"]) not in members
            ):
                missing += 1
            else:
                selected.append(row)
        return selected, missing

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tar_handles"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_tar_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


def _select_split_rows(
    rows: Sequence[Mapping],
    split: str,
    val_fraction: float,
    seed: int,
    min_query_views: int,
) -> List[dict]:
    if split not in {"gallery_train", "gallery_val"}:
        return [dict(row) for row in rows]
    reference, queries = partition_gallery_uploads(
        rows,
        val_fraction=val_fraction,
        seed=seed,
        min_query_views=min_query_views,
    )
    return reference if split == "gallery_train" else queries


def _class_mapping(
    rows: Sequence[Mapping], classes: Optional[Sequence[str]]
) -> Tuple[List[str], Dict[str, int]]:
    if classes is None:
        class_list = sorted({str(row["hotel_id"]) for row in rows})
    else:
        class_list = [str(hotel_id) for hotel_id in classes]
    return class_list, {hotel_id: index for index, hotel_id in enumerate(class_list)}


class OpenHotelsRetrievalTrainDataset(_OpenHotelsImageStore, Dataset):
    """One stochastic multi-view query/positive-gallery episode per hotel.

    The dataset is indexed by eligible hotel rather than source image.  Pair it
    with :class:`HotelBalancedBatchSampler` to produce batches of distinct hotel
    identities.  A sampler index can be either ``hotel_index`` or
    ``(hotel_index, episode_seed)``; the latter makes stochastic episode choices
    deterministic across worker scheduling.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "gallery_train",
        classes: Optional[Sequence[str]] = None,
        min_query_views: int = 2,
        max_query_views: int = 4,
        positive_gallery_size: int = 4,
        val_fraction: float = 0.1,
        seed: int = 0,
        skip_missing: bool = True,
        image_size: int = 224,
        transform=None,
    ) -> None:
        if split not in {"gallery", "gallery_train"}:
            raise ValueError("training split must be 'gallery' or 'gallery_train'")
        if not 1 <= min_query_views <= max_query_views:
            raise ValueError("require 1 <= min_query_views <= max_query_views")
        if positive_gallery_size < 1:
            raise ValueError("positive_gallery_size must be at least 1")

        self._init_store(data_dir)
        self.split = split
        self.seed = seed
        self.min_query_views = min_query_views
        self.max_query_views = max_query_views
        self.positive_gallery_size = positive_gallery_size
        self.transform = transform or build_retrieval_transform(True, image_size)

        all_rows = load_openhotels_metadata(data_dir, split)
        self.classes, self.class_to_idx = _class_mapping(all_rows, classes)
        self.num_classes = len(self.classes)
        rows = _select_split_rows(
            all_rows, split, val_fraction, seed, min_query_views
        )
        rows, self.missing_count = self._available_rows(rows, skip_missing)

        groups: Dict[int, Dict[str, List[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            label = self.class_to_idx.get(str(row["hotel_id"]))
            if label is not None:
                groups[label][_room_key(row)].append(row)

        self.groups_by_label: Dict[int, Dict[str, List[dict]]] = {
            label: dict(room_groups) for label, room_groups in groups.items()
        }
        self.query_groups_by_label: Dict[int, List[str]] = {}
        for label, room_groups in self.groups_by_label.items():
            total_rows = sum(len(group_rows) for group_rows in room_groups.values())
            candidates = []
            for room, group_rows in room_groups.items():
                # Reserve at least one source-disjoint positive after taking the
                # minimum-size coherent query.
                if (
                    len(group_rows) >= min_query_views
                    and total_rows - min_query_views >= 1
                ):
                    candidates.append(room)
            if candidates:
                self.query_groups_by_label[label] = sorted(candidates)

        self.eligible_labels = sorted(self.query_groups_by_label)
        self.excluded_hotel_count = len(self.classes) - len(self.eligible_labels)
        if not self.eligible_labels:
            raise RuntimeError("No hotels can form a source-disjoint retrieval episode")

    def __len__(self) -> int:
        return len(self.eligible_labels)

    @staticmethod
    def _decode_index(index) -> Tuple[int, Optional[int]]:
        if isinstance(index, (tuple, list)):
            if len(index) != 2:
                raise ValueError("episode index must be (hotel_index, seed)")
            return int(index[0]), int(index[1])
        return int(index), None

    def __getitem__(self, index) -> dict:
        hotel_index, episode_seed = self._decode_index(index)
        label = self.eligible_labels[hotel_index]
        if episode_seed is None:
            # torch initial_seed is independently initialized for every loader
            # worker.  The balanced sampler supplies a seed in normal use.
            episode_seed = (
                torch.initial_seed() + hotel_index + random.randrange(2**31)
            ) % (2**63 - 1)
        rng = random.Random(episode_seed)

        room_groups = self.groups_by_label[label]
        query_room = rng.choice(self.query_groups_by_label[label])
        source_rows = room_groups[query_room]
        total_rows = sum(len(group_rows) for group_rows in room_groups.values())
        # Preserve as many distinct positive images as the requested gallery
        # size allows. Small hotels still participate with every source-disjoint
        # positive they have instead of duplicating an image to pad the set.
        reserved_gallery = min(
            self.positive_gallery_size,
            total_rows - self.min_query_views,
        )
        max_views = min(
            self.max_query_views,
            len(source_rows),
            total_rows - reserved_gallery,
        )
        num_views = rng.randint(self.min_query_views, max_views)
        query_rows = rng.sample(source_rows, num_views)
        query_paths = {str(row["path"]) for row in query_rows}

        cross_upload = [
            row
            for room, group_rows in room_groups.items()
            if room != query_room
            for row in group_rows
            if str(row["path"]) not in query_paths
        ]
        same_upload = [
            row for row in source_rows if str(row["path"]) not in query_paths
        ]
        rng.shuffle(cross_upload)
        rng.shuffle(same_upload)
        candidates = cross_upload + same_upload
        gallery_rows = candidates[: self.positive_gallery_size]
        if not gallery_rows:
            raise RuntimeError(
                f"Hotel {self.classes[label]} has no source-disjoint positive"
            )

        query_images = torch.stack([
            self.transform(self._load_image(row)) for row in query_rows
        ])
        gallery_images = torch.stack([
            self.transform(self._load_image(row)) for row in gallery_rows
        ])
        gallery_labels = torch.full(
            (len(gallery_rows),), label, dtype=torch.long
        )
        return {
            "query_images": query_images,
            "query_mask": torch.ones(len(query_rows), dtype=torch.bool),
            "gallery_images": gallery_images,
            "gallery_mask": torch.ones(len(gallery_rows), dtype=torch.bool),
            "query_label": torch.tensor(label, dtype=torch.long),
            "gallery_labels": gallery_labels,
            "hotel_id": self.classes[label],
            "query_group": (self.classes[label], query_room),
            "query_paths": [str(row["path"]) for row in query_rows],
            "gallery_paths": [str(row["path"]) for row in gallery_rows],
            "query_metadata": query_rows,
            "gallery_metadata": gallery_rows,
            "gallery_cross_upload": torch.tensor([
                _room_key(row) != query_room for row in gallery_rows
            ], dtype=torch.bool),
        }


class HotelBalancedBatchSampler(Sampler[List[Tuple[int, int]]]):
    """Sample a fixed number of distinct hotel identities per episode batch."""

    def __init__(
        self,
        dataset: OpenHotelsRetrievalTrainDataset,
        hotels_per_batch: int,
        steps_per_epoch: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if hotels_per_batch < 2:
            raise ValueError("hotels_per_batch must be at least 2 for negatives")
        if hotels_per_batch > len(dataset):
            raise ValueError(
                "hotels_per_batch cannot exceed the number of eligible hotels"
            )
        self.dataset = dataset
        self.hotels_per_batch = hotels_per_batch
        self.full_pass = steps_per_epoch is None
        self.steps_per_epoch = (
            int(steps_per_epoch)
            if steps_per_epoch is not None
            else int(math.ceil(len(dataset) / hotels_per_batch))
        )
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be at least 1")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        population = list(range(len(self.dataset)))
        if self.full_pass:
            # A single shuffled pass gives every eligible hotel one episode per
            # epoch. If the final remainder is a singleton, repeat one hotel as
            # its negative rather than emitting a batch with no negative class.
            rng.shuffle(population)
            batches = [
                population[start:start + self.hotels_per_batch]
                for start in range(0, len(population), self.hotels_per_batch)
            ]
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-1].append(batches[0][0])
        else:
            # Explicit steps_per_epoch is a with-replacement schedule. Each
            # physical batch still contains distinct hotels for valid mining.
            batches = [
                rng.sample(population, self.hotels_per_batch)
                for _ in range(self.steps_per_epoch)
            ]
        for hotel_indices in batches:
            yield [
                (hotel_index, rng.randrange(2**63 - 1))
                for hotel_index in hotel_indices
            ]


def _diverse_rows(
    rows: Sequence[Mapping],
    max_views: int,
    seed: int,
    collection_index: int = 0,
) -> List[dict]:
    """Choose deterministic rows round-robin across view/object categories."""

    if len(rows) <= max_views:
        return sorted((dict(row) for row in rows), key=lambda row: row["path"])

    buckets: Dict[str, List[dict]] = defaultdict(list)
    for original in rows:
        row = dict(original)
        category = row.get("view_type") or row.get("object_type") or "__unknown__"
        buckets[str(category)].append(row)
    salt = seed + collection_index * 97_409
    for category in buckets:
        buckets[category].sort(
            key=lambda row: _stable_order(str(row["path"]), salt)
        )
    categories = sorted(
        buckets,
        key=lambda category: _stable_order(category, salt),
    )

    selected: List[dict] = []
    depth = 0
    while len(selected) < max_views:
        added = False
        for category in categories:
            if depth < len(buckets[category]):
                selected.append(buckets[category][depth])
                added = True
                if len(selected) == max_views:
                    break
        if not added:
            break
        depth += 1
    return selected


class OpenHotelsQueryDataset(_OpenHotelsImageStore, Dataset):
    """Deterministic room/upload-grouped queries for validation or testing."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        classes: Optional[Sequence[str]] = None,
        min_views: int = 2,
        max_views: int = 4,
        collections_per_group: int = 1,
        val_fraction: float = 0.1,
        seed: int = 0,
        skip_missing: bool = True,
        image_size: int = 224,
        transform=None,
    ) -> None:
        if split not in {"gallery_val", "test_room", "test_object"}:
            raise ValueError(
                "query split must be gallery_val, test_room, or test_object"
            )
        if not 1 <= min_views <= max_views:
            raise ValueError("require 1 <= min_views <= max_views")
        if collections_per_group < 1:
            raise ValueError("collections_per_group must be at least 1")

        self._init_store(data_dir)
        self.split = split
        self.min_views = min_views
        self.max_views = max_views
        self.seed = seed
        self.transform = transform or build_retrieval_transform(False, image_size)

        all_rows = load_openhotels_metadata(data_dir, split)
        self.classes, self.class_to_idx = _class_mapping(all_rows, classes)
        self.num_classes = len(self.classes)
        rows = _select_split_rows(
            all_rows, split, val_fraction, seed, min_views
        )
        rows, self.missing_count = self._available_rows(rows, skip_missing)

        grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for row in rows:
            hotel_id = str(row["hotel_id"])
            if hotel_id in self.class_to_idx:
                grouped[(hotel_id, _room_key(row))].append(row)

        self.samples: List[dict] = []
        for (hotel_id, room) in sorted(grouped):
            group_rows = grouped[(hotel_id, room)]
            if len(group_rows) < min_views:
                continue
            seen_path_sets = set()
            for collection_index in range(collections_per_group):
                chosen = _diverse_rows(
                    group_rows,
                    max_views=max_views,
                    seed=seed + _stable_order(f"{hotel_id}:{room}", seed),
                    collection_index=collection_index,
                )
                path_key = tuple(sorted(str(row["path"]) for row in chosen))
                if path_key in seen_path_sets:
                    continue
                seen_path_sets.add(path_key)
                self.samples.append({
                    "hotel_id": hotel_id,
                    "room": room,
                    "rows": chosen,
                })
        if not self.samples:
            raise RuntimeError(f"No valid multi-view queries found for {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        rows = sample["rows"]
        label = self.class_to_idx[sample["hotel_id"]]
        images = torch.stack([
            self.transform(self._load_image(row)) for row in rows
        ])
        return {
            "query_images": images,
            "query_mask": torch.ones(len(rows), dtype=torch.bool),
            "query_label": torch.tensor(label, dtype=torch.long),
            "hotel_id": sample["hotel_id"],
            "query_group": (sample["hotel_id"], sample["room"]),
            "query_paths": [str(row["path"]) for row in rows],
            "query_metadata": rows,
        }


class OpenHotelsGalleryDataset(_OpenHotelsImageStore, Dataset):
    """Flat single-image reference gallery for embedding and FAISS indexing."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "gallery",
        classes: Optional[Sequence[str]] = None,
        val_fraction: float = 0.1,
        val_min_query_views: int = 2,
        seed: int = 0,
        skip_missing: bool = True,
        image_size: int = 224,
        transform=None,
    ) -> None:
        if split not in {"gallery", "gallery_train"}:
            raise ValueError("gallery split must be 'gallery' or 'gallery_train'")
        self._init_store(data_dir)
        self.split = split
        self.transform = transform or build_retrieval_transform(False, image_size)

        all_rows = load_openhotels_metadata(data_dir, split)
        self.classes, self.class_to_idx = _class_mapping(all_rows, classes)
        self.num_classes = len(self.classes)
        rows = _select_split_rows(
            all_rows, split, val_fraction, seed, val_min_query_views
        )
        rows, self.missing_count = self._available_rows(rows, skip_missing)
        self.rows = [
            row for row in rows
            if str(row["hotel_id"]) in self.class_to_idx
        ]
        self.rows.sort(key=lambda row: str(row["path"]))
        if not self.rows:
            raise RuntimeError(f"No available gallery images found for {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        hotel_id = str(row["hotel_id"])
        return {
            "image": self.transform(self._load_image(row)),
            "label": torch.tensor(self.class_to_idx[hotel_id], dtype=torch.long),
            "hotel_id": hotel_id,
            "path": str(row["path"]),
            "metadata": row,
        }


class OpenHotelsFlatQueryDataset(_OpenHotelsImageStore, Dataset):
    """Every official query image as an independent retrieval query.

    This dataset complements :class:`OpenHotelsQueryDataset`, which forms
    room/upload-level multi-view queries and therefore omits singleton uploads.
    It uses the same item schema as :class:`OpenHotelsGalleryDataset`, allowing
    ``gallery_collate_fn`` to serve as the flat-query collator as well.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        classes: Optional[Sequence[str]] = None,
        skip_missing: bool = True,
        image_size: int = 224,
        transform=None,
    ) -> None:
        if split not in {"test_room", "test_object"}:
            raise ValueError("flat query split must be test_room or test_object")
        self._init_store(data_dir)
        self.split = split
        self.transform = transform or build_retrieval_transform(False, image_size)

        all_rows = load_openhotels_metadata(data_dir, split)
        self.classes, self.class_to_idx = _class_mapping(all_rows, classes)
        self.num_classes = len(self.classes)
        rows, self.missing_count = self._available_rows(all_rows, skip_missing)
        self.rows = [
            row for row in rows
            if str(row["hotel_id"]) in self.class_to_idx
        ]
        self.rows.sort(key=lambda row: str(row["path"]))
        if not self.rows:
            raise RuntimeError(f"No available flat query images found for {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        hotel_id = str(row["hotel_id"])
        return {
            "image": self.transform(self._load_image(row)),
            "label": torch.tensor(self.class_to_idx[hotel_id], dtype=torch.long),
            "hotel_id": hotel_id,
            "path": str(row["path"]),
            "metadata": row,
        }


def _pad_image_sets(batch: Sequence[Mapping], key: str):
    tensors = [item[key] for item in batch]
    max_count = max(tensor.shape[0] for tensor in tensors)
    shape = (len(tensors), max_count) + tuple(tensors[0].shape[1:])
    padded = tensors[0].new_zeros(shape)
    mask = torch.zeros((len(tensors), max_count), dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        if tuple(tensor.shape[1:]) != tuple(tensors[0].shape[1:]):
            raise ValueError(f"inconsistent image shapes for {key}")
        count = tensor.shape[0]
        padded[index, :count] = tensor
        mask[index, :count] = True
    return padded, mask


def retrieval_collate_fn(batch: Sequence[Mapping]) -> dict:
    """Pad variable query/positive image sets and preserve path metadata."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    query_images, query_mask = _pad_image_sets(batch, "query_images")
    result = {
        "query_images": query_images,
        "query_mask": query_mask,
        "query_label": torch.stack([item["query_label"] for item in batch]),
        "hotel_id": [item["hotel_id"] for item in batch],
        "query_group": [item["query_group"] for item in batch],
        "query_paths": [item["query_paths"] for item in batch],
        "query_metadata": [item["query_metadata"] for item in batch],
    }
    has_gallery = ["gallery_images" in item for item in batch]
    if any(has_gallery) and not all(has_gallery):
        raise ValueError("cannot mix query-only and training episodes")
    if all(has_gallery):
        gallery_images, gallery_mask = _pad_image_sets(batch, "gallery_images")
        labels = torch.full(gallery_mask.shape, -1, dtype=torch.long)
        cross_upload = torch.zeros(gallery_mask.shape, dtype=torch.bool)
        for index, item in enumerate(batch):
            count = item["gallery_labels"].shape[0]
            labels[index, :count] = item["gallery_labels"]
            cross_upload[index, :count] = item["gallery_cross_upload"]
        result.update({
            "gallery_images": gallery_images,
            "gallery_mask": gallery_mask,
            "gallery_labels": labels,
            "gallery_paths": [item["gallery_paths"] for item in batch],
            "gallery_metadata": [item["gallery_metadata"] for item in batch],
            "gallery_cross_upload": cross_upload,
        })
    return result


def gallery_collate_fn(batch: Sequence[Mapping]) -> dict:
    """Collate the flat gallery without recursively collating nullable metadata."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "hotel_id": [item["hotel_id"] for item in batch],
        "paths": [item["path"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def build_openhotels_seen_validation(
    data_dir: str | Path,
    classes: Optional[Sequence[str]] = None,
    val_fraction: float = 0.1,
    min_views: int = 2,
    max_views: int = 4,
    seed: int = 0,
    skip_missing: bool = True,
    image_size: int = 224,
) -> Tuple[OpenHotelsGalleryDataset, OpenHotelsQueryDataset]:
    """Create a matched upload-disjoint gallery/query validation pair."""

    gallery = OpenHotelsGalleryDataset(
        data_dir,
        split="gallery_train",
        classes=classes,
        val_fraction=val_fraction,
        val_min_query_views=min_views,
        seed=seed,
        skip_missing=skip_missing,
        image_size=image_size,
    )
    queries = OpenHotelsQueryDataset(
        data_dir,
        split="gallery_val",
        classes=gallery.classes,
        min_views=min_views,
        max_views=max_views,
        val_fraction=val_fraction,
        seed=seed,
        skip_missing=skip_missing,
        image_size=image_size,
    )
    return gallery, queries


__all__ = [
    "HotelBalancedBatchSampler",
    "OpenHotelsGalleryDataset",
    "OpenHotelsFlatQueryDataset",
    "OpenHotelsQueryDataset",
    "OpenHotelsRetrievalTrainDataset",
    "build_openhotels_seen_validation",
    "build_retrieval_transform",
    "gallery_collate_fn",
    "load_openhotels_metadata",
    "partition_gallery_uploads",
    "retrieval_collate_fn",
]
