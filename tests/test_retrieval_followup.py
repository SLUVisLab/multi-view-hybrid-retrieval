import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.check_retrieval_evaluation import check_and_evaluate, compare_direct_results, direct_metrics


class RetrievalFollowupTest(unittest.TestCase):

    def test_unfinished_run_does_not_start_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "monitor.status").write_text("time=test state=running training_pid=1\n")
            args = SimpleNamespace(current_run=root, previous_run=root,
                                   output_dir=root / "check", device="cpu", check_only=False)
            with patch("scripts.check_retrieval_evaluation.direct_metrics") as score:
                report = check_and_evaluate(args, logging.getLogger("test"))
            score.assert_not_called()
            self.assertEqual(report["status"], "not_finished")
            self.assertTrue((root / "check/comparison.json").is_file())

    def test_completed_run_keeps_original_results_and_scores_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, current = root / "previous", root / "current"
            for run in (previous, current):
                run.mkdir()
                (run / "monitor.status").write_text("time=test state=exited:0 training_pid=1\n")
                (run / "retrieval_test_metrics.json").write_text('{"original": true}\n')
            originals = [(run / "retrieval_test_metrics.json").read_bytes() for run in (previous, current)]
            before, after = self.result(20, 0.1), self.result(30, 0.2)
            args = SimpleNamespace(current_run=current, previous_run=previous,
                                   output_dir=root / "check", device="cpu", check_only=False)
            with (
                patch("scripts.check_retrieval_evaluation.evaluation_settings", return_value={}),
                patch("scripts.check_retrieval_evaluation.direct_metrics", side_effect=[before, after]) as score,
            ):
                report = check_and_evaluate(args, logging.getLogger("test"))
            self.assertEqual(score.call_count, 2)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["comparison"][0]["change_percentage_points"]["recall@1"], 10.0)
            for run, original in zip((previous, current), originals):
                self.assertEqual((run / "retrieval_test_metrics.json").read_bytes(), original)
            saved = json.loads((root / "check/comparison.json").read_text())
            self.assertEqual(saved["original_results"]["current"], {"original": True})
            self.assertTrue((root / "check/epoch-020-direct.json").is_file())
            self.assertTrue((root / "check/epoch-030-direct.json").is_file())

    def result(self, epoch, recall):
        scores = {"image": {"joint": {"recall@1": recall}}, "num_queries": 1,
                  "num_gallery_images": 3, "num_gallery_hotels": 2}
        return {"epoch": epoch, "settings": {"seed": 0}, "metrics": {
            split: {protocol: scores.copy() for protocol in ("official_single_image", "multi_view")}
            for split in ("test_room", "test_object")
        }}

    def test_comparison_rejects_different_query_counts_and_sampling(self):
        before, after = self.result(20, 0.1), self.result(30, 0.2)
        after["metrics"]["test_room"]["multi_view"]["num_queries"] = 2
        with self.assertRaisesRegex(ValueError, "counts differ"):
            compare_direct_results(before, after)
        after = self.result(30, 0.2)
        after["settings"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "different query/gallery settings"):
            compare_direct_results(before, after)

    def test_cached_pass_scores_fused_embedding_without_image_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"epoch": 20}, root / "best.pth")
            gallery = {"version": 1, "kind": "gallery",
                       "embeddings": torch.tensor([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]]),
                       "labels": torch.tensor([20, 20, 10])}
            single = {"version": 1, "kind": "gallery", "embeddings": torch.tensor([[0.0, 1.0]]),
                      "labels": torch.tensor([10])}
            query = {"version": 1, "kind": "query", "joint_embeddings": torch.tensor([[1.0, 0.5]]),
                     "labels": torch.tensor([10])}
            torch.save(gallery, root / "gallery-test.pt")
            original = {}
            for split in ("test_room", "test_object"):
                torch.save(query, root / f"{split}-test.pt")
                torch.save(single, root / f"{split}-single-test.pt")
                original[split] = {
                    protocol: {"num_queries": 1, "num_gallery_images": 3}
                    for protocol in ("official_single_image", "multi_view")
                }
            (root / "retrieval_test_metrics.json").write_text(json.dumps(original))
            settings = {"embedding_cache_dir": str(root), "recall_at": (1, 2, 3),
                        "retrieval_query_chunk_size": 1, "retrieval_gallery_chunk_size": 2}
            with patch("retrieval_main._embedding_cache_tag", return_value="test"):
                scored = direct_metrics(root, settings, "cpu", logging.getLogger("test"))
            self.assertEqual(scored["epoch"], 20)
            for split in ("test_room", "test_object"):
                multi = scored["metrics"][split]["multi_view"]
                self.assertEqual(multi["image"]["joint"]["recall@2"], 0.0)
                self.assertEqual(multi["image"]["joint"]["recall@3"], 1.0)
                self.assertNotIn("hotel", multi)
                flat = scored["metrics"][split]["official_single_image"]
                self.assertEqual(flat["image"]["joint"]["recall@1"], 1.0)

            tag = "1234567890abcdef"
            for path in root.glob("*-test.pt"):
                path.rename(path.with_name(path.name.replace("-test.pt", f"-{tag}.pt")))
            with patch("retrieval_main._embedding_cache_tag", side_effect=AssertionError("Checkpoint reread")):
                scored = direct_metrics(root, settings, "cpu", logging.getLogger("test"),
                                        cache_tag=tag, checkpoint_epoch=20)
            self.assertEqual(scored["epoch"], 20)
            self.assertEqual(scored["metrics"]["test_room"]["multi_view"]["image"]["joint"]["recall@3"], 1.0)

    def test_missing_cache_fails_before_scoring_or_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = {"embedding_cache_dir": directory}
            with (
                patch("retrieval_main._embedding_cache_tag", return_value="missing"),
                patch("torch.load") as load,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "Required embedding caches missing"):
                    direct_metrics(directory, settings, "cpu", logging.getLogger("test"))
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
