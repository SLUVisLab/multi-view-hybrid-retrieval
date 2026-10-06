"""Easy Positive metric-learning losses for asymmetric retrieval.

The implementation follows the mining definitions from *Improved Embeddings
with Easy Positive Triplet Mining* while allowing query and gallery embeddings
to be different tensors.  This is useful for multi-view queries whose positives
are single-image gallery embeddings.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrResult = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


class EasyPositiveLoss(nn.Module):
    """Easy Positive loss with all, hard, or semi-hard negatives.

    For every query, the easiest positive is the gallery item with the same
    label and largest cosine similarity. The negative term is selected by
    ``mode``:

    * ``ep``: all gallery negatives (the paper's EP / NCA formulation)
    * ``ephn``: the most similar negative
    * ``epshn``: the most similar negative that is less similar than the
      selected positive

    Args:
        mode: One of ``ep``, ``ephn``, or ``epshn``. ``ep_all`` is accepted as
            an alias for ``ep``.
        temperature: Softmax temperature. The paper uses 0.1.
        normalize: L2-normalize query and gallery embeddings before mining.
        reduction: ``mean``, ``sum``, or ``none``.
        semi_hard_fallback: EPSHN behavior when a query has no semi-hard
            negative. ``hardest`` uses its hard negative so the query still
            trains; ``skip`` omits that query from the reduced loss.
        exclude_self: Whether to remove the query/gallery diagonal from the
            positive mask. If None, exclusion is enabled automatically when
            query and gallery are the same tensor.
        ignore_index: Labels with this value do not participate.

    ``positive_mask`` and ``negative_mask`` in :meth:`forward` are additional
    boolean constraints. For example, use ``positive_mask`` to prevent a query
    source image from serving as its own gallery positive.
    """

    _VALID_MODES = {'ep', 'ephn', 'epshn'}

    def __init__(self, mode='epshn', temperature=0.1, normalize=True,
                 reduction='mean', semi_hard_fallback='hardest',
                 exclude_self=None, ignore_index=-1):
        super().__init__()
        mode = mode.lower().replace('-', '_')
        if mode == 'ep_all':
            mode = 'ep'
        if mode not in self._VALID_MODES:
            raise ValueError(
                f'mode must be one of {sorted(self._VALID_MODES)}, got {mode!r}')
        if temperature <= 0:
            raise ValueError('temperature must be greater than zero')
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        if semi_hard_fallback not in {'hardest', 'skip'}:
            raise ValueError("semi_hard_fallback must be 'hardest' or 'skip'")
        if exclude_self not in {None, True, False}:
            raise ValueError('exclude_self must be None, True, or False')

        self.mode = mode
        self.temperature = float(temperature)
        self.normalize = normalize
        self.reduction = reduction
        self.semi_hard_fallback = semi_hard_fallback
        self.exclude_self = exclude_self
        self.ignore_index = ignore_index

    @staticmethod
    def _check_mask(mask, name, shape, device):
        if mask is None:
            return None
        if tuple(mask.shape) != shape:
            raise ValueError(
                f'{name} must have shape {shape}, got {tuple(mask.shape)}')
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _masked_max(values, mask):
        masked = values.masked_fill(~mask, -torch.inf)
        maxima, indices = masked.max(dim=1)
        return maxima, indices, mask.any(dim=1)

    def _reduce(self, per_query_loss, valid_query, similarities):
        valid_losses = per_query_loss[valid_query]
        if self.reduction == 'none':
            return per_query_loss
        if valid_losses.numel() == 0:
            # A differentiable zero is friendlier to sparse/partially padded
            # episodic batches than mean(empty), which yields NaN.
            return similarities.sum() * 0.0
        if self.reduction == 'sum':
            return valid_losses.sum()
        return valid_losses.mean()

    def forward(self, query_embeddings, gallery_embeddings, query_labels,
                gallery_labels, positive_mask=None, negative_mask=None,
                return_details=False) -> TensorOrResult:
        """Compute a query-to-gallery Easy Positive loss.

        Embeddings have shapes ``[num_queries, dim]`` and
        ``[num_gallery, dim]``; labels are one-dimensional. The returned scalar
        is differentiable with respect to both query and selected gallery
        embeddings.
        """

        if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
            raise ValueError('query_embeddings and gallery_embeddings must be 2-D')
        if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
            raise ValueError(
                'Query and gallery embedding dimensions differ: '
                f'{query_embeddings.shape[1]} vs {gallery_embeddings.shape[1]}')

        query_labels = query_labels.reshape(-1).to(query_embeddings.device)
        gallery_labels = gallery_labels.reshape(-1).to(query_embeddings.device)
        if query_labels.numel() != query_embeddings.shape[0]:
            raise ValueError('query_labels length must match query_embeddings')
        if gallery_labels.numel() != gallery_embeddings.shape[0]:
            raise ValueError('gallery_labels length must match gallery_embeddings')
        if gallery_embeddings.device != query_embeddings.device:
            raise ValueError('Query and gallery embeddings must be on the same device')

        num_queries = query_embeddings.shape[0]
        num_gallery = gallery_embeddings.shape[0]
        mask_shape = (num_queries, num_gallery)
        positive_mask = self._check_mask(
            positive_mask, 'positive_mask', mask_shape, query_embeddings.device)
        negative_mask = self._check_mask(
            negative_mask, 'negative_mask', mask_shape, query_embeddings.device)

        same_storage = (
            query_embeddings.shape == gallery_embeddings.shape and
            query_embeddings.data_ptr() == gallery_embeddings.data_ptr())

        # Keep mining and loss arithmetic in float32 under mixed precision.
        # Autocast matmul otherwise returns bfloat16 while softplus returns
        # float32, which cannot be assigned into a bfloat16 loss buffer.
        with torch.autocast(device_type=query_embeddings.device.type, enabled=False):
            if query_embeddings.dtype in (torch.float16, torch.bfloat16):
                query_embeddings = query_embeddings.float()
            if gallery_embeddings.dtype in (torch.float16, torch.bfloat16):
                gallery_embeddings = gallery_embeddings.float()
            if self.normalize:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)
            similarities = query_embeddings @ gallery_embeddings.transpose(0, 1)

        valid_query_label = query_labels != self.ignore_index
        valid_gallery_label = gallery_labels != self.ignore_index
        same_label = query_labels[:, None].eq(gallery_labels[None, :])
        positive_candidates = (
            same_label & valid_query_label[:, None] & valid_gallery_label[None, :])
        negative_candidates = (
            ~same_label & valid_query_label[:, None] & valid_gallery_label[None, :])
        if positive_mask is not None:
            positive_candidates &= positive_mask
        if negative_mask is not None:
            negative_candidates &= negative_mask

        exclude_self = self.exclude_self
        if exclude_self is None:
            exclude_self = same_storage
        if exclude_self:
            if num_queries != num_gallery:
                raise ValueError(
                    'exclude_self=True requires equally sized query and gallery sets')
            positive_candidates.fill_diagonal_(False)

        positive_similarity, positive_index, has_positive = self._masked_max(
            similarities, positive_candidates)
        hard_negative_similarity, hard_negative_index, has_negative = self._masked_max(
            similarities, negative_candidates)
        valid_query = valid_query_label & has_positive & has_negative

        selected_negative_similarity = hard_negative_similarity
        selected_negative_index = hard_negative_index
        has_semi_hard = torch.zeros_like(valid_query)
        if self.mode == 'epshn':
            # Mining decisions do not need gradients. Detaching the threshold
            # avoids retaining a discontinuous comparison in the autograd graph.
            semi_hard_candidates = (
                negative_candidates &
                (similarities.detach() < positive_similarity.detach()[:, None]))
            semi_similarity, semi_index, has_semi_hard = self._masked_max(
                similarities, semi_hard_candidates)
            if self.semi_hard_fallback == 'hardest':
                selected_negative_similarity = torch.where(
                    has_semi_hard, semi_similarity, hard_negative_similarity)
                selected_negative_index = torch.where(
                    has_semi_hard, semi_index, hard_negative_index)
            else:
                selected_negative_similarity = semi_similarity
                selected_negative_index = semi_index
                valid_query &= has_semi_hard

        per_query_loss = similarities.new_zeros(num_queries)
        if torch.any(valid_query):
            positive_logits = positive_similarity[valid_query] / self.temperature
            if self.mode == 'ep':
                negative_logits = similarities[valid_query] / self.temperature
                negative_logits = negative_logits.masked_fill(
                    ~negative_candidates[valid_query], -torch.inf)
                denominator = torch.logsumexp(
                    torch.cat((positive_logits[:, None], negative_logits), dim=1),
                    dim=1)
                valid_losses = denominator - positive_logits
            else:
                negative_logits = (
                    selected_negative_similarity[valid_query] / self.temperature)
                valid_losses = F.softplus(negative_logits - positive_logits)
            per_query_loss[valid_query] = valid_losses

        loss = self._reduce(per_query_loss, valid_query, similarities)
        if not return_details:
            return loss

        details = {
            'positive_similarity': positive_similarity.detach(),
            'negative_similarity': selected_negative_similarity.detach(),
            'positive_index': positive_index.detach(),
            'negative_index': selected_negative_index.detach(),
            'valid_query_mask': valid_query.detach(),
            'has_semi_hard': has_semi_hard.detach(),
            'num_valid': valid_query.sum().detach(),
            'num_queries': torch.as_tensor(
                num_queries, device=query_embeddings.device),
            'loss_per_query': per_query_loss.detach(),
        }
        return loss, details


class EPAllLoss(EasyPositiveLoss):
    """Easy Positive loss against all negative gallery items."""

    def __init__(self, **kwargs):
        super().__init__(mode='ep', **kwargs)


class EPHNLoss(EasyPositiveLoss):
    """Easy Positive Hard Negative loss."""

    def __init__(self, **kwargs):
        super().__init__(mode='ephn', **kwargs)


class EPSHNLoss(EasyPositiveLoss):
    """Easy Positive Semi-Hard Negative loss."""

    def __init__(self, **kwargs):
        super().__init__(mode='epshn', **kwargs)
