"""Training loop for asymmetric multi-view-to-gallery retrieval."""

import datetime
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def _branch_embedding(outputs, branch):
    values = outputs[branch]
    for key in ("embedding", "embeddings", "features"):
        if key in values:
            return values[key]
    raise KeyError(f"{branch!r} output does not contain an embedding")


def _classification_weight(initial_weight, decay_epochs, epoch):
    if initial_weight <= 0:
        return 0.0
    if decay_epochs <= 0:
        return initial_weight
    progress = min(max((epoch - 1) / decay_epochs, 0.0), 1.0)
    return initial_weight * (1.0 - progress)


def build_retrieval_scheduler(optimizer, max_lr, epochs, steps_per_epoch,
                             completed_epochs=0, additional_epochs=0,
                             scheduler_phase=None):
    """Map a resumed schedule to its loader, or start a new continuation phase."""
    if additional_epochs < 0:
        raise ValueError("additional_epochs cannot be negative")
    if completed_epochs < 0 or (not additional_epochs and completed_epochs > epochs):
        raise ValueError("completed epochs must lie between zero and num_epochs")
    if additional_epochs:
        # Optimization state is retained, but the phase has its own full cycle.
        # Training still receives global epochs for sampling and classifier decay.
        scheduler_phase = {
            "start_epoch": completed_epochs + 1,
            "num_epochs": additional_epochs,
        }
        epochs = additional_epochs
        completed_epochs = 0
    elif scheduler_phase is not None:
        if (not isinstance(scheduler_phase, dict)
                or not {"start_epoch", "num_epochs"}.issubset(scheduler_phase)
                or any(isinstance(scheduler_phase[key], bool)
                       or not isinstance(scheduler_phase[key], int)
                       or scheduler_phase[key] < 1
                       for key in ("start_epoch", "num_epochs"))):
            raise ValueError("resume checkpoint has invalid scheduler_phase")
        phase_start = scheduler_phase["start_epoch"]
        phase_epochs = scheduler_phase["num_epochs"]
        phase_final = phase_start + phase_epochs - 1
        if epochs != phase_final:
            raise ValueError(
                f"num_epochs must match resumed phase final epoch ({phase_final}); "
                "use --additional_epochs to start a new phase"
            )
        completed_epochs -= phase_start - 1
        if not 0 <= completed_epochs <= phase_epochs:
            raise ValueError("checkpoint epoch lies outside its scheduler phase")
        epochs = phase_epochs
    options = {
        "max_lr": max_lr,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "div_factor": 10,
        "final_div_factor": 1000,
        "pct_start": min(0.3, max(1 / epochs, 5 / epochs)),
        "anneal_strategy": "cos",
    }
    # Initialize the parameter-group schedule settings from this run's config.
    # Resumption then maps completed epochs onto the new number of batch steps,
    # rather than loading an old total_steps that no longer matches the loader.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **options)
    if completed_epochs:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, **options,
            last_epoch=completed_epochs * steps_per_epoch - 1,
        )
    scheduler.retrieval_scheduler_phase = scheduler_phase
    return scheduler


class RetrievalTrainer:
    """Optimize a fused query against source-disjoint gallery images.

    The dataloader must yield the dictionary produced by
    ``retrieval_collate_fn``. Mining happens inside each physical batch;
    gradient accumulation only changes optimizer-step frequency.
    """

    def __init__(self, model, criterion, optimizer, device, logger, save_dir,
                 scheduler=None, view_loss_weight=0.25,
                 classification_loss_weight=0.1,
                 classification_decay_epochs=5, grad_accum_steps=1,
                 grad_clip_norm=None, log_interval=20, checkpoint_metadata=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.save_dir = Path(save_dir)
        self.scheduler = scheduler
        self.scheduler_phase = None
        self.view_loss_weight = view_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.classification_decay_epochs = classification_decay_epochs
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _unpack_loss(result):
        if isinstance(result, tuple):
            return result
        return result, {}

    @staticmethod
    def _summarize_mining(details):
        summary = {}
        valid = details.get("valid_query_mask")
        if torch.is_tensor(valid):
            valid = valid.bool()
            summary["num_valid"] = valid.sum()
            for source, destination in (
                    ("positive_similarity", "positive_similarity"),
                    ("negative_similarity", "negative_similarity")):
                values = details.get(source)
                if torch.is_tensor(values) and torch.any(valid):
                    summary[destination] = values[valid].mean()
            semi_hard = details.get("has_semi_hard")
            if torch.is_tensor(semi_hard) and torch.any(valid):
                summary["semi_hard_fraction"] = semi_hard[valid].float().mean()
        elif "num_valid" in details:
            summary["num_valid"] = details["num_valid"]
        return summary

    def _forward_batch(self, batch, epoch):
        query_images = batch["query_images"].to(self.device, non_blocking=True)
        query_mask = batch["query_mask"].to(self.device, non_blocking=True)
        query_labels = batch["query_label"].to(self.device, non_blocking=True)
        gallery_images = batch["gallery_images"].to(self.device, non_blocking=True)
        gallery_mask = batch["gallery_mask"].to(self.device, non_blocking=True)
        gallery_labels = batch["gallery_labels"].to(self.device, non_blocking=True)

        classification_weight = _classification_weight(
            self.classification_loss_weight,
            self.classification_decay_epochs,
            epoch,
        )
        need_logits = classification_weight > 0

        query_outputs = self.model(
            query_images, view_mask=query_mask, return_embeddings=True,
            return_logits=need_logits,
        )

        valid_gallery_images = gallery_images[gallery_mask]
        valid_gallery_labels = gallery_labels[gallery_mask]
        gallery_outputs = self.model(
            valid_gallery_images.unsqueeze(1), return_embeddings=True,
            return_logits=need_logits, return_collection=False,
        )

        joint_embeddings = _branch_embedding(query_outputs, "mv_collection")
        gallery_embeddings = _branch_embedding(gallery_outputs, "single")
        joint_loss, joint_details = self._unpack_loss(self.criterion(
            joint_embeddings, gallery_embeddings, query_labels,
            valid_gallery_labels, return_details=True,
        ))

        B, N = query_mask.shape
        view_embeddings = _branch_embedding(query_outputs, "single")
        if view_embeddings.ndim == 2:
            view_embeddings = view_embeddings.reshape(B, N, -1)
        valid_view_embeddings = view_embeddings[query_mask]
        valid_view_labels = query_labels[:, None].expand(B, N)[query_mask]
        view_loss, view_details = self._unpack_loss(self.criterion(
            valid_view_embeddings, gallery_embeddings, valid_view_labels,
            valid_gallery_labels, return_details=True,
        ))

        classification_loss = joint_loss.new_zeros(())
        if need_logits:
            joint_logits = query_outputs["mv_collection"]["logits"]
            view_logits = query_outputs["single"]["logits"]
            if view_logits.ndim == 2:
                view_logits = view_logits.reshape(B, N, -1)
            gallery_logits = gallery_outputs["single"]["logits"]
            classification_loss = (
                F.cross_entropy(joint_logits, query_labels)
                + F.cross_entropy(view_logits[query_mask], valid_view_labels)
                + F.cross_entropy(gallery_logits, valid_gallery_labels)
            ) / 3.0

        total_loss = (
            joint_loss
            + self.view_loss_weight * view_loss
            + classification_weight * classification_loss
        )
        values = {
            "total": total_loss.detach(),
            "joint": joint_loss.detach(),
            "view": view_loss.detach(),
            "classification": classification_loss.detach(),
            "classification_weight": classification_weight,
        }
        for prefix, details in (("joint", joint_details), ("view", view_details)):
            for key, value in self._summarize_mining(details).items():
                if torch.is_tensor(value) and value.numel() == 1:
                    values[f"{prefix}_{key}"] = value.detach()
                elif isinstance(value, (float, int)):
                    values[f"{prefix}_{key}"] = value
        return total_loss, values

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        sampler = getattr(dataloader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        self.optimizer.zero_grad(set_to_none=True)
        started = datetime.datetime.now()
        window = {}
        window_batches = 0
        num_batches = len(dataloader)

        for batch_index, batch in enumerate(dataloader):
            with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda"):
                loss, values = self._forward_batch(batch, epoch)
                scaled_loss = loss / self.grad_accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite retrieval loss at epoch {epoch}, "
                    f"batch {batch_index + 1}: {float(loss.detach())}"
                )
            scaled_loss.backward()

            should_step = (
                (batch_index + 1) % self.grad_accum_steps == 0
                or batch_index + 1 == num_batches
            )
            if should_step:
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            for key, value in values.items():
                number = float(value) if not torch.is_tensor(value) else float(value.item())
                window[key] = window.get(key, 0.0) + number
            window_batches += 1

            if ((batch_index + 1) % self.log_interval == 0
                    or batch_index + 1 == num_batches):
                averages = {key: value / window_batches for key, value in window.items()}
                lr = self.optimizer.param_groups[0]["lr"]
                extras = " ".join(
                    f"{key}={value:.4f}" for key, value in sorted(averages.items())
                    if key not in {"total", "joint", "view", "classification",
                                   "classification_weight"}
                )
                self.logger.info(
                    "Epoch %d batch %d/%d loss total=%.4f joint=%.4f "
                    "view=%.4f cls=%.4f cls_weight=%.4f lr=%.6g %s",
                    epoch, batch_index + 1, num_batches,
                    averages["total"], averages["joint"], averages["view"],
                    averages["classification"], averages["classification_weight"],
                    lr, extras,
                )
                window = {}
                window_batches = 0

        self.logger.info("Epoch %d training took %s", epoch,
                         datetime.datetime.now() - started)

    def save_checkpoint(self, path, epoch, best_score, metrics=None):
        path = Path(path)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "metrics": metrics or {},
        }
        if "classes" in self.checkpoint_metadata:
            payload["classes"] = list(self.checkpoint_metadata["classes"])
        if "config" in self.checkpoint_metadata:
            payload["config"] = dict(self.checkpoint_metadata["config"])
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
            scheduler_phase = getattr(self.scheduler, "retrieval_scheduler_phase", None)
            if scheduler_phase is not None:
                payload["scheduler_phase"] = dict(scheduler_phase)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        self.logger.info("Saved retrieval checkpoint to %s", path)

    def restore_training_state(self, checkpoint):
        """Restore optimization progress after the model checkpoint is loaded."""
        if not isinstance(checkpoint, dict) or not {
                "model", "optimizer", "epoch"}.issubset(checkpoint):
            raise ValueError("--resume requires a retrieval training checkpoint")
        epoch = checkpoint["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("resume checkpoint epoch must be a non-negative integer")
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler_phase = checkpoint.get("scheduler_phase")
        return epoch + 1, float(checkpoint.get("best_score", float("-inf")))

    def fit(self, dataloader, epochs, validate_fn=None, eval_every=1,
            start_epoch=1, best_score=float("-inf")):
        if not 1 <= start_epoch <= epochs + 1:
            raise ValueError("start_epoch must lie between 1 and num_epochs + 1")
        best_path = self.save_dir / "best.pth"
        last_path = self.save_dir / "last.pth"
        saved_best_this_run = False
        resumed_best_available = (
            start_epoch > 1 and validate_fn is not None
            and best_score > float("-inf") and best_path.is_file()
        )
        for epoch in range(start_epoch, epochs + 1):
            self.train_epoch(dataloader, epoch)
            metrics = {}
            score = None
            if (validate_fn is not None
                    and (epoch % eval_every == 0 or epoch == epochs)):
                metrics, score = validate_fn(self.model, epoch)
                self.logger.info("Epoch %d retrieval validation: %s", epoch, metrics)
            if score is not None and score > best_score:
                best_score = score
                self.save_checkpoint(best_path, epoch, best_score, metrics)
                saved_best_this_run = True
            self.save_checkpoint(last_path, epoch, best_score, metrics)

        # A resumed run can retain its previously validated best model. Fresh
        # runs and runs without validation must replace any unrelated best.pth.
        if not saved_best_this_run and not resumed_best_available:
            self.save_checkpoint(best_path, epochs, best_score, {})
        return best_path
