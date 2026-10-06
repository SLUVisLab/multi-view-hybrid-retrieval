import logging
import tempfile
import unittest
from pathlib import Path

import torch

from engine.retrieval_trainer import (
    RetrievalTrainer,
    _classification_weight,
    build_retrieval_scheduler,
)


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class _NoOpTrainer(RetrievalTrainer):
    def train_epoch(self, dataloader, epoch):
        self.trained_epochs.append(epoch)


class _OptimizerStepTrainer(_NoOpTrainer):
    def train_epoch(self, dataloader, epoch):
        super().train_epoch(dataloader, epoch)
        self.classification_weights.append(_classification_weight(
            self.classification_loss_weight,
            self.classification_decay_epochs,
            epoch,
        ))
        for batch in dataloader:
            self.optimizer.zero_grad(set_to_none=True)
            self.model(batch).square().mean().backward()
            self.optimizer.step()
            self.scheduler.step()


class RetrievalTrainerTest(unittest.TestCase):

    def make_trainer(self, save_dir, trainer_class=_NoOpTrainer):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        trainer = trainer_class(
            model=model,
            criterion=None,
            optimizer=optimizer,
            device=torch.device("cpu"),
            logger=logging.getLogger("retrieval-trainer-test"),
            save_dir=save_dir,
        )
        trainer.trained_epochs = []
        return trainer

    def test_no_validation_overwrites_stale_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            save_dir = Path(directory)
            torch.save({"epoch": -99}, save_dir / "best.pth")
            trainer = self.make_trainer(save_dir)
            best_path = trainer.fit([], epochs=1, validate_fn=None)
            self.assertEqual(_load(best_path)["epoch"], 1)

    def test_resume_restores_optimizer_momentum_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.model(torch.ones(1, 2)).sum().backward()
            trainer.optimizer.step()
            trainer.save_checkpoint(Path(directory) / "last.pth", 3, 0.8)
            checkpoint = _load(Path(directory) / "last.pth")
            resumed = self.make_trainer(directory)
            resumed.model.load_state_dict(checkpoint["model"])
            start_epoch, best_score = resumed.restore_training_state(checkpoint)
            self.assertEqual(start_epoch, 4)
            self.assertEqual(best_score, 0.8)
            for original, restored in zip(
                    trainer.optimizer.state.values(), resumed.optimizer.state.values()):
                torch.testing.assert_close(
                    restored["momentum_buffer"], original["momentum_buffer"],
                )
            resumed.fit([], epochs=5, start_epoch=start_epoch, best_score=best_score)
            self.assertEqual(resumed.trained_epochs, [4, 5])
            final = _load(Path(directory) / "last.pth")
            self.assertEqual(final["epoch"], 5)
            self.assertEqual(final["best_score"], 0.8)

    def test_resume_keeps_previous_validated_best_if_scores_do_not_improve(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            best_path = Path(directory) / "best.pth"
            trainer.save_checkpoint(best_path, 3, 0.8, {"recall": 0.8})
            original_bytes = best_path.read_bytes()
            trainer.fit(
                [], epochs=5, start_epoch=4, best_score=0.8,
                validate_fn=lambda model, epoch: ({"recall": 0.7}, 0.7),
            )
            self.assertEqual(trainer.trained_epochs, [4, 5])
            self.assertEqual(best_path.read_bytes(), original_bytes)
            self.assertEqual(_load(Path(directory) / "last.pth")["best_score"], 0.8)

    def test_resume_replaces_best_when_score_improves(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.save_checkpoint(Path(directory) / "best.pth", 3, 0.8)
            trainer.fit(
                [], epochs=4, start_epoch=4, best_score=0.8,
                validate_fn=lambda model, epoch: ({"recall": 0.9}, 0.9),
            )
            best = _load(Path(directory) / "best.pth")
            self.assertEqual(best["epoch"], 4)
            self.assertEqual(best["best_score"], 0.9)

    def test_resume_before_first_validation_overwrites_unrelated_best(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            torch.save({"epoch": -99}, Path(directory) / "best.pth")
            trainer.fit(
                [], epochs=2, start_epoch=2,
                validate_fn=lambda model, epoch: ({"recall": 0.7}, 0.7),
                eval_every=5,
            )
            self.assertEqual(_load(Path(directory) / "best.pth")["epoch"], 2)

    def test_resume_rejects_initialization_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            with self.assertRaisesRegex(ValueError, "retrieval training checkpoint"):
                trainer.restore_training_state({"model": trainer.model.state_dict()})

    def test_scheduler_maps_completed_epochs_to_new_batch_steps(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        old_scheduler = build_retrieval_scheduler(
            optimizer, max_lr=0.1, epochs=20, steps_per_epoch=10,
        )
        for _ in range(10):
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            old_scheduler.step()
        saved_optimizer = optimizer.state_dict()
        momentum_before = {
            key: value["momentum_buffer"].clone()
            for key, value in saved_optimizer["state"].items()
        }

        resumed_model = torch.nn.Linear(2, 2)
        resumed_optimizer = torch.optim.SGD(
            resumed_model.parameters(), lr=0.1, momentum=0.9,
        )
        resumed_optimizer.load_state_dict(saved_optimizer)
        resumed_scheduler = build_retrieval_scheduler(
            resumed_optimizer, max_lr=0.1, epochs=20, steps_per_epoch=4,
            completed_epochs=1,
        )
        self.assertEqual(resumed_scheduler.last_epoch, 4)
        self.assertEqual(resumed_scheduler.total_steps, 80)
        for key, value in resumed_optimizer.state_dict()["state"].items():
            torch.testing.assert_close(value["momentum_buffer"], momentum_before[key])

        reference_model = torch.nn.Linear(2, 2)
        reference_optimizer = torch.optim.SGD(
            reference_model.parameters(), lr=0.1, momentum=0.9,
        )
        reference_scheduler = build_retrieval_scheduler(
            reference_optimizer, max_lr=0.1, epochs=20, steps_per_epoch=4,
        )
        for _ in range(4):
            reference_optimizer.step()
            reference_scheduler.step()
        self.assertAlmostEqual(
            resumed_optimizer.param_groups[0]["lr"],
            reference_optimizer.param_groups[0]["lr"],
        )
        self.assertAlmostEqual(
            resumed_optimizer.param_groups[0]["momentum"],
            reference_optimizer.param_groups[0]["momentum"],
        )
        self.assertGreater(resumed_optimizer.param_groups[0]["lr"], 0.1 / 10)
        for _ in range(19 * 4):
            resumed_optimizer.step()
            resumed_scheduler.step()
        self.assertEqual(resumed_scheduler.last_epoch, resumed_scheduler.total_steps)

    def test_scheduler_rejects_checkpoint_beyond_requested_epochs(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        with self.assertRaisesRegex(ValueError, "completed epochs"):
            build_retrieval_scheduler(
                optimizer, max_lr=0.1, epochs=2, steps_per_epoch=4,
                completed_epochs=3,
            )

    def test_completed_schedule_starts_new_phase_and_keeps_global_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.model(torch.ones(1, 2)).sum().backward()
            trainer.optimizer.step()
            trainer.scheduler = build_retrieval_scheduler(
                trainer.optimizer, max_lr=0.1, epochs=20, steps_per_epoch=10,
                completed_epochs=20,
            )
            checkpoint_path = Path(directory) / "last.pth"
            trainer.save_checkpoint(checkpoint_path, 20, 0.3345)
            best_path = Path(directory) / "best.pth"
            trainer.save_checkpoint(best_path, 20, 0.3345, {"recall": 0.3345})
            original_best = best_path.read_bytes()
            checkpoint = _load(checkpoint_path)
            resumed = self.make_trainer(directory, _OptimizerStepTrainer)
            resumed.classification_weights = []
            resumed.model.load_state_dict(checkpoint["model"])
            start_epoch, best_score = resumed.restore_training_state(checkpoint)
            momentum_before = {
                key: value["momentum_buffer"].clone()
                for key, value in resumed.optimizer.state_dict()["state"].items()
            }
            resumed.scheduler = build_retrieval_scheduler(
                resumed.optimizer, max_lr=0.2, epochs=20, steps_per_epoch=4,
                completed_epochs=start_epoch - 1, additional_epochs=20,
            )
            self.assertEqual(start_epoch, 21)
            self.assertEqual(best_score, 0.3345)
            self.assertEqual(resumed.scheduler.last_epoch, 0)
            self.assertEqual(resumed.scheduler.total_steps, 80)
            self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], 0.02)
            for key, value in resumed.optimizer.state_dict()["state"].items():
                torch.testing.assert_close(value["momentum_buffer"], momentum_before[key])

            resumed.fit(
                [torch.ones(1, 2)] * 4, epochs=40, start_epoch=start_epoch,
                best_score=best_score, eval_every=5,
                validate_fn=lambda model, epoch: ({"recall": 0.3}, 0.3),
            )
            self.assertEqual(resumed.trained_epochs, list(range(21, 41)))
            self.assertEqual(resumed.classification_weights, [0.0] * 20)
            self.assertEqual(resumed.scheduler.last_epoch, 80)
            final = _load(checkpoint_path)
            self.assertEqual(final["epoch"], 40)
            self.assertEqual(final["best_score"], 0.3345)
            self.assertEqual(final["scheduler"]["last_epoch"], 80)
            self.assertEqual(final["scheduler_phase"], {
                "start_epoch": 21, "num_epochs": 20,
            })
            self.assertEqual(best_path.read_bytes(), original_best)

    def test_midphase_resume_maps_local_epochs_after_batch_size_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, _OptimizerStepTrainer)
            trainer.classification_weights = []
            trainer.scheduler = build_retrieval_scheduler(
                trainer.optimizer, max_lr=0.1, epochs=20, steps_per_epoch=10,
                completed_epochs=20, additional_epochs=20,
            )
            for epoch in range(21, 26):
                trainer.train_epoch([torch.ones(1, 2)] * 10, epoch)
            checkpoint_path = Path(directory) / "last.pth"
            trainer.save_checkpoint(checkpoint_path, 25, 0.3345)
            checkpoint = _load(checkpoint_path)
            resumed = self.make_trainer(directory, _OptimizerStepTrainer)
            resumed.classification_weights = []
            resumed.model.load_state_dict(checkpoint["model"])
            start_epoch, best_score = resumed.restore_training_state(checkpoint)
            self.assertEqual(resumed.scheduler_phase, {
                "start_epoch": 21, "num_epochs": 20,
            })
            resumed.scheduler = build_retrieval_scheduler(
                resumed.optimizer, max_lr=0.1, epochs=40, steps_per_epoch=4,
                completed_epochs=start_epoch - 1,
                scheduler_phase=resumed.scheduler_phase,
            )
            self.assertEqual(start_epoch, 26)
            self.assertEqual(resumed.scheduler.last_epoch, 20)
            self.assertEqual(resumed.scheduler.total_steps, 80)
            reference_model = torch.nn.Linear(2, 2)
            reference_optimizer = torch.optim.SGD(
                reference_model.parameters(), lr=0.1, momentum=0.9,
            )
            reference_scheduler = build_retrieval_scheduler(
                reference_optimizer, max_lr=0.1, epochs=20, steps_per_epoch=4,
            )
            for _ in range(20):
                reference_optimizer.step()
                reference_scheduler.step()
            self.assertAlmostEqual(
                resumed.optimizer.param_groups[0]["lr"],
                reference_optimizer.param_groups[0]["lr"],
            )
            self.assertAlmostEqual(
                resumed.optimizer.param_groups[0]["momentum"],
                reference_optimizer.param_groups[0]["momentum"],
            )
            resumed.fit(
                [torch.ones(1, 2)] * 4, epochs=40,
                start_epoch=start_epoch, best_score=best_score,
            )
            self.assertEqual(resumed.trained_epochs, list(range(26, 41)))
            self.assertEqual(resumed.scheduler.last_epoch, 80)
            final = _load(checkpoint_path)
            self.assertEqual(final["epoch"], 40)
            self.assertEqual(final["scheduler_phase"], resumed.scheduler_phase)

    def test_midphase_resume_rejects_conflicting_global_final_epoch(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        with self.assertRaisesRegex(ValueError, "resumed phase final epoch"):
            build_retrieval_scheduler(
                optimizer, max_lr=0.1, epochs=30, steps_per_epoch=4,
                completed_epochs=25,
                scheduler_phase={"start_epoch": 21, "num_epochs": 20},
            )

    def test_new_phase_rejects_negative_duration(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        with self.assertRaisesRegex(ValueError, "additional_epochs"):
            build_retrieval_scheduler(
                optimizer, max_lr=0.1, epochs=20, steps_per_epoch=4,
                completed_epochs=20, additional_epochs=-1,
            )


if __name__ == "__main__":
    unittest.main()
