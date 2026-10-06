import logging
import tempfile
import unittest
from pathlib import Path

import torch

from engine.retrieval_trainer import RetrievalTrainer
from retrieval_main import _load_checkpoint


class RetrievalCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("retrieval-checkpoint-test")
        self.model = torch.nn.Linear(2, 2)

    def test_metadata_round_trip_and_incompatible_label_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            config = {"dataset": "openhotels", "model_max_views": 4}
            trainer = RetrievalTrainer(
                self.model, None, torch.optim.SGD(self.model.parameters(), lr=0.1),
                torch.device("cpu"), self.logger, directory,
                checkpoint_metadata={"classes": ["a", "b"], "config": config},
            )
            trainer.save_checkpoint(path, epoch=1, best_score=0.5)
            clone = torch.nn.Linear(2, 2)
            payload = _load_checkpoint(clone, path, self.logger, ["a", "b"], config)
            self.assertEqual(payload["classes"], ["a", "b"])
            self.assertEqual(payload["config"], config)
            torch.testing.assert_close(clone.weight, self.model.weight)
            with self.assertRaisesRegex(ValueError, "class order"):
                _load_checkpoint(clone, path, self.logger, ["b", "a"], config)
            with self.assertRaisesRegex(ValueError, "model_max_views"):
                _load_checkpoint(clone, path, self.logger, ["a", "b"],
                                 {"model_max_views": 2})

    def test_legacy_checkpoint_without_metadata_still_loads_strictly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            torch.save({"model": self.model.state_dict()}, path)
            clone = torch.nn.Linear(2, 2)
            _load_checkpoint(clone, path, self.logger, ["a", "b"],
                             {"model_max_views": 4})
            torch.testing.assert_close(clone.weight, self.model.weight)
            incompatible = torch.nn.Linear(2, 3)
            with self.assertRaises(RuntimeError):
                _load_checkpoint(incompatible, path, self.logger)


if __name__ == "__main__":
    unittest.main()
