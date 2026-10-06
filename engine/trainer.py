import datetime

import einops
import torch

from .engine import EngineBase


def cur_step(cur_epoch, idx, N, fmt=None):
    _cur_step = cur_epoch + idx / N
    if fmt:
        return fmt.format(_cur_step)
    else:
        return _cur_step


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class TrainerEngine(EngineBase):

    def _train_epoch(self, dataloader, cur_epoch):

        self.model.train()
        train_acc = {}

        self.optimizer_model.zero_grad(set_to_none=True)
        num_batches = len(dataloader)
        window_loss = {'total': 0., 'single': 0., 'mv_collection': 0., 'md': 0.}
        window_batches = 0
        for idx, (images, targets, _) in enumerate(dataloader):
            B, N, _, _, _ = images.shape
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == 'cuda'):
                output = self.model(images)
                total_loss = torch.tensor(0.).to(self.device, non_blocking=True)
                batch_loss = {}

                for view_type in output:

                    if view_type == 'mv_collection':
                        t = targets[:, 0].flatten()
                    elif view_type == 'single':
                        t = targets.flatten()
                    logits = output[view_type]['logits']
                    base_loss = self.criterion(logits, t)
                    batch_loss[view_type] = base_loss
                    total_loss += base_loss

                    pred = torch.argmax(logits, dim=1)
                    if view_type not in train_acc:
                        train_acc[view_type] = {'total': 0, 'correct': 0}
                    pred = pred[t != -1]
                    t = t[t != -1]
                    train_acc[view_type]['correct'] += (pred == t).long().sum().item()
                    train_acc[view_type]['total'] += t.numel()

                md_value = total_loss.new_zeros(())
                if self.md_loss is not None:
                    single_view_logits = einops.rearrange(output['single']['logits'], '(b n) k -> b n k', b=B, n=N)
                    md_value = self.md_loss(output['mv_collection']['logits'], single_view_logits, targets[:, 0])
                    total_loss += md_value

                window_loss['total'] += total_loss.detach()
                window_loss['single'] += batch_loss['single'].detach()
                if 'mv_collection' in batch_loss:
                    window_loss['mv_collection'] += batch_loss['mv_collection'].detach()
                window_loss['md'] += md_value.detach()
                window_batches += 1

                (total_loss / self.grad_accum_steps).backward()
                should_step = ((idx + 1) % self.grad_accum_steps == 0 or
                               idx + 1 == num_batches)
                if should_step:
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer_model.step()
                    self.optimizer_model.zero_grad(set_to_none=True)
                    if self.lr_scheduler_model is not None:
                        try:
                            self.lr_scheduler_model.step()
                        except ValueError:
                            pass
            if (idx + 1) % self.log_interval == 0 or idx + 1 == num_batches:
                averages = {
                    key: float(value / window_batches)
                    for key, value in window_loss.items()
                }
                self.logger.info(
                    'Epoch %d batch %d/%d (%.1f%%) loss total=%.4f '
                    'single=%.4f collection=%.4f md=%.4f lr=%.6g',
                    cur_epoch, idx + 1, num_batches,
                    100 * (idx + 1) / num_batches,
                    averages['total'], averages['single'],
                    averages['mv_collection'], averages['md'],
                    get_lr(self.optimizer_model))
                window_loss = {'total': 0., 'single': 0.,
                               'mv_collection': 0., 'md': 0.}
                window_batches = 0

        for view_type in train_acc:
            if train_acc[view_type]['total'] > 0:
                acc = train_acc[view_type]['correct'] / train_acc[view_type]['total']
                self.logger.info(f'Epoch {cur_epoch}, {view_type} Train top1_acc: {acc}')

    def _eval_epoch(self, dataloader, cur_epoch, model_save_to, best_model_save_to, best_acc):

        scores = self.evaluate(dataloader)
        self.metadata['scores'] = scores
        save_key = 'mv_collection' if 'mv_collection' in scores else 'single'
        
        save_metric = 'top1_acc'
        save_score = scores[save_key][save_metric]

        if best_acc < save_score:
            best_acc = save_score
            self.metadata['best_score'] = best_acc
            self.metadata['best_epoch'] = cur_epoch
            self.save_models(best_model_save_to, self.metadata)

        for view_type in scores:
            for metric in scores[view_type]:
                self.logger.info(f'Epoch {cur_epoch}, {view_type} Validation {metric}: {scores[view_type][metric]}')

        self.save_models(model_save_to, self.metadata)
        return best_acc

    def train(self, tr_loader, n_epochs, val_loader):

        model_save_to = f'{self.save_dir}/last.pth'
        best_model_save_to = f'{self.save_dir}/best.pth'

        dt = datetime.datetime.now()
        self.model_to_device()
        # Ensure a checkpoint exists even when a large-class validation pass has
        # zero top-1 accuracy at initialization.
        best_acc = float('-inf')
        for cur_epoch in range(1, n_epochs + 1):
            if cur_epoch == 1 and not self.skip_initial_eval:
                best_acc = self._eval_epoch(val_loader, 0, model_save_to, best_model_save_to, best_acc)

            self._train_epoch(tr_loader, cur_epoch)
            self.metadata['cur_epoch'] = cur_epoch
            self.metadata['lr'] = get_lr(self.optimizer_model)
            should_evaluate = cur_epoch % self.eval_every == 0 or cur_epoch == n_epochs
            if should_evaluate:
                best_acc = self._eval_epoch(val_loader, cur_epoch, model_save_to, best_model_save_to, best_acc)
            else:
                self.save_models(model_save_to, self.metadata)

            elapsed = datetime.datetime.now() - dt
            expected_total = elapsed / cur_epoch * n_epochs
            expected_remain = expected_total - elapsed
            self.logger.info('expected remain {}'.format(expected_remain))
        self.logger.info('finish engine, takes {}'.format(datetime.datetime.now() - dt))
