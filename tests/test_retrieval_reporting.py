import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import retrieval_main
from engine.retrieval_evaluator import RetrievalEvaluator


class RetrievalReportingTest(unittest.TestCase):

    def test_official_report_includes_direct_single_and_fused_image_recall(self):
        # Two incorrect-hotel images outrank the correct image for the fused
        # query. Hotel deduplication and the hybrid blend improve different
        # cutoffs, so neither can substitute for the direct image ranking.
        gallery = {
            "embeddings": torch.tensor([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]]),
            "labels": torch.tensor([20, 20, 10]),
        }
        queries = {
            "joint_embeddings": torch.tensor([[1.0, 0.5]]),
            "view_embeddings": torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
            "view_mask": torch.tensor([[True, True]]),
            "labels": torch.tensor([10]),
        }
        single = {
            "embeddings": torch.tensor([[0.0, 1.0]]),
            "labels": torch.tensor([10]),
        }
        evaluator = RetrievalEvaluator(device="cpu", use_amp=False)
        evaluator.extract_gallery = Mock(side_effect=[gallery, single, single])
        evaluator.extract_queries = Mock(return_value=queries)
        logger = Mock()

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                data_dir=directory, save_dir=directory, seed=0,
                skip_missing=False, image_size=16, num_workers=0,
                gallery_batch_size=3, query_batch_size=1, eval_log_interval=1,
                embedding_cache_dir=None, reuse_embedding_cache=False,
                test_min_views=2, max_query_views=4,
                test_collections_per_group=1, max_test_queries=0,
                max_test_image_queries=0, recall_at=(1, 2, 3),
                hotel_top_m=(1,), hybrid_alpha=0.5,
                retrieval_query_chunk_size=1, retrieval_gallery_chunk_size=2,
            )
            with (
                patch.object(retrieval_main, "OpenHotelsGalleryDataset", return_value=[]),
                patch.object(retrieval_main, "OpenHotelsQueryDataset", return_value=[]),
                patch.object(retrieval_main, "OpenHotelsFlatQueryDataset", return_value=[]),
                patch.object(retrieval_main, "DataLoader", return_value=[]),
                patch.object(retrieval_main, "RetrievalEvaluator", return_value=evaluator),
            ):
                result = retrieval_main.run_official_evaluation(
                    None, [], args, torch.device("cpu"), logger, "test",
                )

            saved = json.loads((Path(directory) / "retrieval_test_metrics.json").read_text())
            self.assertEqual(saved, result)
            for split in ("test_room", "test_object"):
                multi = saved[split]["multi_view"]
                self.assertEqual(multi["image"]["joint"], {
                    "recall@1": 0.0, "recall@2": 0.0, "recall@3": 1.0,
                })
                self.assertEqual(multi["hotel"]["max"]["joint"]["recall@2"], 1.0)
                self.assertEqual(multi["hotel"]["max"]["hybrid"]["recall@1"], 1.0)
                flat = saved[split]["official_single_image"]
                self.assertEqual(flat["image"]["joint"]["recall@1"], 1.0)
                self.assertNotIn("hotel", flat)

        messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertEqual(sum("single-image embedding to gallery images" in m for m in messages), 2)
        self.assertEqual(sum("fused multi-image embedding to gallery images" in m for m in messages), 2)


if __name__ == "__main__":
    unittest.main()
