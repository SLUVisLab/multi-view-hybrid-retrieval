import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from datasets.openhotels_retrieval import (
    HotelBalancedBatchSampler,
    OpenHotelsFlatQueryDataset,
    OpenHotelsQueryDataset,
    OpenHotelsRetrievalTrainDataset,
    build_openhotels_seen_validation,
    gallery_collate_fn,
    retrieval_collate_fn,
)


class OpenHotelsRetrievalDataTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        gallery = []

        def add_gallery(hotel, room, count, offset=0):
            for index in range(count):
                gallery.append({
                    "path": f"images/gallery/{hotel}/{room}-{index}.jpg",
                    "hotel_id": hotel,
                    "room": room,
                    "timestamp": "2025-01-01T00:00:00",
                    "is_object": False,
                    "view_type": ["bedroom", "bathroom", "living area"][
                        (index + offset) % 3
                    ],
                    "object_type": None,
                    "shard": "shards/gallery-00000.tar",
                })

        add_gallery("A", "a1", 5)
        add_gallery("A", "a2", 3, 1)
        add_gallery("B", "b1", 3)
        add_gallery("B", "b2", 1)
        add_gallery("C", "c1", 1)  # Cannot make a query or positive.
        add_gallery("D", "d1", 2)  # Query would exhaust its only upload.

        test_room = []
        categories = [
            "bedroom", "bathroom", "living area", "kitchen", "closet",
            "balcony",
        ]
        for index, category in enumerate(categories):
            test_room.append({
                "path": f"images/test_room/A/{index}.jpg",
                "hotel_id": "A",
                "room": "test-upload",
                "timestamp": "2025-02-01T00:00:00",
                "is_object": False,
                "view_type": category,
                "object_type": None,
                "shard": "shards/test_room-00000.tar",
            })

        self._write_split("metadata_gallery.json", gallery)
        self._write_split("metadata_test_room.json", test_room)
        self._write_split("metadata_test_object.json", [])
        for number, row in enumerate(gallery + test_room):
            image_path = (
                self.root / "images" / Path(row["shard"]).stem / row["path"]
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB", (24, 24), color=(number % 255, 20, 40)
            ).save(image_path)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_split(self, filename, rows):
        with (self.root / filename).open("w") as handle:
            json.dump(rows, handle)

    def test_episode_prefers_cross_upload_and_never_reuses_source(self):
        dataset = OpenHotelsRetrievalTrainDataset(
            self.root,
            split="gallery",
            min_query_views=2,
            max_query_views=4,
            positive_gallery_size=4,
            seed=5,
            image_size=16,
        )
        self.assertEqual(len(dataset), 2)  # Only hotels A and B are eligible.
        for hotel_index in range(len(dataset)):
            episode = dataset[(hotel_index, 1234 + hotel_index)]
            self.assertGreaterEqual(len(episode["query_paths"]), 2)
            self.assertLessEqual(len(episode["query_paths"]), 4)
            self.assertTrue(
                set(episode["query_paths"]).isdisjoint(episode["gallery_paths"])
            )
            # If another upload exists, its rows occur before same-upload
            # fallback rows in the positive set.
            flags = episode["gallery_cross_upload"].tolist()
            self.assertTrue(flags[0])
            if False in flags:
                first_fallback = flags.index(False)
                self.assertTrue(all(flags[:first_fallback]))
                self.assertTrue(not any(flags[first_fallback:]))

    def test_balanced_sampler_and_variable_set_collation(self):
        dataset = OpenHotelsRetrievalTrainDataset(
            self.root, split="gallery", seed=9, image_size=16
        )
        sampler = HotelBalancedBatchSampler(
            dataset, hotels_per_batch=2, steps_per_epoch=3, seed=11
        )
        first_pass = list(sampler)
        self.assertEqual(len(first_pass), 3)
        for batch_indices in first_pass:
            self.assertEqual(len({item[0] for item in batch_indices}), 2)
        episodes = [dataset[index] for index in first_pass[0]]
        batch = retrieval_collate_fn(episodes)
        self.assertEqual(batch["query_images"].shape[0], 2)
        self.assertEqual(batch["query_images"].shape[2:], (3, 16, 16))
        self.assertTrue(torch.equal(
            batch["query_mask"].sum(1),
            torch.tensor([len(item["query_paths"]) for item in episodes]),
        ))
        self.assertTrue(torch.equal(
            batch["gallery_mask"].sum(1),
            torch.tensor([len(item["gallery_paths"]) for item in episodes]),
        ))
        self.assertTrue((batch["gallery_labels"][~batch["gallery_mask"]] == -1).all())

    def test_default_balanced_sampler_covers_every_hotel(self):
        class FiveHotels:
            def __len__(self):
                return 5

        sampler = HotelBalancedBatchSampler(
            FiveHotels(), hotels_per_batch=2, seed=31,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(len(batch) >= 2 for batch in batches))
        self.assertTrue(all(
            len({hotel_index for hotel_index, _ in batch}) == len(batch)
            for batch in batches
        ))
        covered = {
            hotel_index for batch in batches for hotel_index, _ in batch
        }
        self.assertEqual(covered, set(range(5)))

    def test_seen_validation_holds_out_complete_uploads(self):
        gallery, queries = build_openhotels_seen_validation(
            self.root, val_fraction=0.5, seed=17, image_size=16
        )
        gallery_groups = {
            (row["hotel_id"], row["room"]) for row in gallery.rows
        }
        query_groups = {
            (sample["hotel_id"], sample["room"]) for sample in queries.samples
        }
        self.assertTrue(gallery_groups.isdisjoint(query_groups))
        self.assertEqual({sample["hotel_id"] for sample in queries.samples}, {"A", "B"})
        gallery_paths = {row["path"] for row in gallery.rows}
        query_paths = {
            row["path"]
            for sample in queries.samples
            for row in sample["rows"]
        }
        self.assertTrue(gallery_paths.isdisjoint(query_paths))

    def test_eval_selection_is_deterministic_and_category_diverse(self):
        first = OpenHotelsQueryDataset(
            self.root,
            split="test_room",
            classes=["A", "B", "C", "D"],
            max_views=4,
            seed=23,
            image_size=16,
        )
        second = OpenHotelsQueryDataset(
            self.root,
            split="test_room",
            classes=["A", "B", "C", "D"],
            max_views=4,
            seed=23,
            image_size=16,
        )
        first_paths = [row["path"] for row in first.samples[0]["rows"]]
        second_paths = [row["path"] for row in second.samples[0]["rows"]]
        self.assertEqual(first_paths, second_paths)
        categories = {
            row["view_type"] for row in first.samples[0]["rows"]
        }
        self.assertEqual(len(first_paths), 4)
        self.assertEqual(len(categories), 4)

    def test_flat_query_keeps_every_official_test_image(self):
        dataset = OpenHotelsFlatQueryDataset(
            self.root,
            split="test_room",
            classes=["A", "B", "C", "D"],
            image_size=16,
        )
        self.assertEqual(len(dataset), 6)
        batch = gallery_collate_fn([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["images"].shape), (2, 3, 16, 16))
        self.assertEqual(tuple(batch["labels"].shape), (2,))
        self.assertEqual(len(batch["paths"]), 2)


if __name__ == "__main__":
    unittest.main()
