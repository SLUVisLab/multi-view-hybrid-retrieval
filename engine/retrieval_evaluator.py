"""Memory-bounded evaluation for multi-view hotel retrieval.

The primary entry point is :meth:`RetrievalEvaluator.evaluate_embeddings`.  It
compares a collection-level embedding and/or its constituent view embeddings
against a gallery of *image* embeddings.  Image similarities are reduced to a
hotel score using either the maximum similarity or the mean of the top-m
similarities for that hotel.

Only one query chunk and one gallery chunk are scored at a time.  The evaluator
therefore never materializes the (potentially enormous) complete query x image
or query x hotel matrix.  All reductions are exact; ``top_m`` does not refer to
an approximate candidate shortlist.
"""

from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


_EMBEDDING_KEYS = ("embedding", "embeddings", "feature", "features")
_CACHE_VERSION = 1


def _first(mapping: Mapping[str, Any], names: Sequence[str], default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _as_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), p=2, dim=-1)


def _load_cache(path: os.PathLike, kind: str) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # ``weights_only`` was added after older supported torch releases.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding cache {path} is not a dictionary")
    if payload.get("version") != _CACHE_VERSION or payload.get("kind") != kind:
        raise ValueError(
            f"Embedding cache {path} has incompatible kind/version "
            f"({payload.get('kind')!r}, {payload.get('version')!r})"
        )
    return payload


def _save_cache(path: os.PathLike, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A partial checkpoint should never masquerade as a usable embedding cache.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name,
                                     suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _embedding_from_output(output: Any, branch_names: Sequence[str]) -> Optional[torch.Tensor]:
    """Find an embedding in a model output without ever falling back to logits."""
    if not isinstance(output, Mapping):
        return None
    branch = _first(output, branch_names)
    if torch.is_tensor(branch):
        return branch
    if isinstance(branch, Mapping):
        value = _first(branch, _EMBEDDING_KEYS)
        if torch.is_tensor(value):
            return value
    # Some embedding-only models return flat keys such as ``joint_embedding``.
    for branch_name in branch_names:
        for suffix in _EMBEDDING_KEYS:
            value = output.get(f"{branch_name}_{suffix}")
            if torch.is_tensor(value):
                return value
    return None


def _batch_paths(paths: Any, batch_size: int, num_views: int):
    """Convert default-collated paths to a batch-major list of lists."""
    if paths is None:
        return [[None] * num_views for _ in range(batch_size)]
    if isinstance(paths, (str, Path)):
        return [[str(paths)]]
    paths = list(paths)
    if not paths:
        return [[None] * num_views for _ in range(batch_size)]
    if all(isinstance(path, (str, Path)) for path in paths):
        if num_views == 1 and len(paths) == batch_size:
            return [[str(path)] for path in paths]
        if batch_size == 1 and len(paths) == num_views:
            return [[str(path) for path in paths]]

    # The retrieval collate function deliberately preserves batch-major paths
    # as lists, whereas torch default_collate turns per-sample path lists into
    # a view-major list of tuples. The container type resolves square B == V
    # batches where shape alone would otherwise be ambiguous.
    batch_major_hint = (
        len(paths) == batch_size and all(isinstance(row, list) for row in paths)
    )
    view_major_hint = (
        len(paths) == num_views and all(isinstance(row, tuple) for row in paths)
    )
    rows = [list(row) if not isinstance(row, (str, Path)) else [row]
            for row in paths]
    if batch_major_hint and all(len(row) <= num_views for row in rows):
        return [
            [None if path is None else str(path) for path in row]
            + [None] * (num_views - len(row))
            for row in rows
        ]
    # torch's default collate transposes ``sample -> list[view path]`` into
    # ``view -> tuple[batch path]``. Prefer that convention in an ambiguous
    # square batch because it is what the repository's datasets produce.
    if (view_major_hint or not batch_major_hint) and len(rows) == num_views and all(
        len(row) == batch_size for row in rows
    ):
        return [[None if rows[view][sample] is None else str(rows[view][sample])
                 for view in range(num_views)]
                for sample in range(batch_size)]
    if len(rows) == batch_size and all(len(row) == num_views for row in rows):
        return [[None if path is None else str(path) for path in row] for row in rows]
    raise ValueError(
        f"Could not interpret paths with outer length {len(rows)} as "
        f"batch_size={batch_size}, num_views={num_views}"
    )


def _labels_by_view(labels: Any, batch_size: int, num_views: int) -> torch.Tensor:
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.ndim == 0:
        labels = labels.repeat(batch_size)
    if labels.ndim == 1:
        if labels.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} labels, received {labels.numel()}")
        return labels[:, None].expand(-1, num_views)
    labels = labels.reshape(batch_size, -1)
    if labels.shape[1] == 1:
        return labels.expand(-1, num_views)
    if labels.shape[1] != num_views:
        raise ValueError(
            f"Expected one or {num_views} labels per sample, got {labels.shape[1]}"
        )
    return labels


def _view_mask(mask: Any, batch_size: int, num_views: int) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, num_views, dtype=torch.bool)
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.ndim == 1:
        if num_views == 1 and mask.numel() == batch_size:
            return mask[:, None].cpu()
        if batch_size == 1 and mask.numel() == num_views:
            return mask[None, :].cpu()
    mask = mask.reshape(batch_size, -1)
    if mask.shape[1] != num_views:
        raise ValueError(f"Expected a [{batch_size}, {num_views}] view mask, got {mask.shape}")
    return mask.cpu()


def _unpack_batch(batch: Any, kind: str):
    """Read tuple batches and both evaluation and paired-training dictionaries."""
    if isinstance(batch, Mapping):
        if kind == "query":
            images = _first(batch, ("query_images", "images", "image"))
            labels = _first(batch, ("query_labels", "query_label", "labels", "label",
                                    "targets", "target"))
            paths = _first(batch, ("query_paths", "query_path", "paths", "path"))
            mask = _first(batch, ("query_mask", "view_mask", "mask"))
        else:
            images = _first(batch, ("gallery_images", "images", "image"))
            labels = _first(batch, ("gallery_labels", "gallery_label", "labels", "label",
                                    "targets", "target"))
            paths = _first(batch, ("gallery_paths", "gallery_path", "paths", "path"))
            mask = _first(batch, ("gallery_mask", "view_mask", "mask"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[:2]
        paths = batch[2] if len(batch) > 2 else None
        mask = batch[3] if len(batch) > 3 else None
    else:
        raise TypeError("A retrieval batch must be a mapping or a tuple (images, labels, ...)")
    if images is None or labels is None:
        raise KeyError(f"The {kind} batch does not contain images and labels")
    return images, labels, paths, mask


def exact_cosine_topk(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    k: int,
    *,
    device: Optional[torch.device] = None,
    query_chunk_size: int = 256,
    gallery_chunk_size: int = 32768,
    backend: str = "torch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return exact cosine nearest neighbours without a full similarity matrix.

    ``backend='faiss'`` uses ``faiss.IndexFlatIP`` when the optional dependency is
    installed. It remains an exact search and runs on CPU. The torch backend can
    use either CPU or GPU and chunks both axes.
    """
    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError("query_embeddings and gallery_embeddings must both be 2-D")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("Query and gallery embedding dimensions differ")
    if gallery_embeddings.shape[0] == 0:
        raise ValueError("The gallery is empty")
    if k < 1:
        raise ValueError("k must be positive")
    k = min(k, gallery_embeddings.shape[0])

    queries = _normalize(query_embeddings).cpu()
    gallery = _normalize(gallery_embeddings).cpu()
    if backend == "faiss":
        try:
            import faiss
        except ImportError as error:
            raise ImportError(
                "backend='faiss' requires the optional faiss package; use backend='torch'"
            ) from error
        index = faiss.IndexFlatIP(gallery.shape[1])
        index.add(gallery.numpy())
        all_scores, all_indices = [], []
        for start in range(0, len(queries), query_chunk_size):
            scores, indices = index.search(
                queries[start:start + query_chunk_size].numpy(), k
            )
            all_scores.append(torch.from_numpy(scores))
            all_indices.append(torch.from_numpy(indices).long())
        return torch.cat(all_scores), torch.cat(all_indices)
    if backend != "torch":
        raise ValueError("backend must be 'torch' or 'faiss'")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_scores, all_indices = [], []
    for query_start in range(0, len(queries), query_chunk_size):
        query = queries[query_start:query_start + query_chunk_size].to(device)
        best_scores = torch.full((len(query), k), -torch.inf, device=device)
        best_indices = torch.full((len(query), k), -1, dtype=torch.long, device=device)
        for gallery_start in range(0, len(gallery), gallery_chunk_size):
            gallery_chunk = gallery[gallery_start:gallery_start + gallery_chunk_size].to(device)
            similarities = query @ gallery_chunk.T
            local_k = min(k, gallery_chunk.shape[0])
            local_scores, local_indices = similarities.topk(local_k, dim=1)
            local_indices += gallery_start
            candidate_scores = torch.cat((best_scores, local_scores), dim=1)
            candidate_indices = torch.cat((best_indices, local_indices), dim=1)
            best_scores, selected = candidate_scores.topk(k, dim=1)
            best_indices = candidate_indices.gather(1, selected)
        all_scores.append(best_scores.cpu())
        all_indices.append(best_indices.cpu())
    return torch.cat(all_scores), torch.cat(all_indices)


class RetrievalEvaluator:
    """Extract and evaluate image-gallery/multi-view-query embeddings."""

    def __init__(self, model=None, device=None, logger=None, log_interval: int = 100,
                 use_amp: bool = True):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger = logger
        self.log_interval = log_interval
        self.use_amp = use_amp

    def _log(self, message: str, *args) -> None:
        if self.logger is not None:
            self.logger.info(message, *args)

    def _forward_embeddings(self, images: torch.Tensor, mask: Optional[torch.Tensor],
                            return_collection: bool = True):
        if self.model is None:
            raise ValueError("A model is required for embedding extraction")
        images = images.to(self.device, non_blocking=True)
        if images.ndim == 4:
            images = images[:, None]
        if images.ndim != 5:
            raise ValueError(f"Expected images shaped [B,V,C,H,W], got {images.shape}")
        device_mask = None if mask is None else mask.to(self.device, non_blocking=True)

        embedding_forward = getattr(self.model, "forward_embeddings", None)
        forward = embedding_forward if callable(embedding_forward) else self.model
        kwargs = {}
        try:
            signature = inspect.signature(
                embedding_forward if callable(embedding_forward) else self.model.forward
            )
            if device_mask is not None and (
                "view_mask" in signature.parameters or
                any(parameter.kind == parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values())
            ):
                kwargs["view_mask"] = device_mask
            if not callable(embedding_forward):
                if "return_embeddings" in signature.parameters:
                    kwargs["return_embeddings"] = True
                if "return_logits" in signature.parameters:
                    kwargs["return_logits"] = False
            if "return_collection" in signature.parameters:
                kwargs["return_collection"] = return_collection
        except (TypeError, ValueError):
            pass

        amp_enabled = self.use_amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            output = forward(images, **kwargs)
        single = _embedding_from_output(output, ("single", "views", "view"))
        joint = _embedding_from_output(
            output, ("mv_collection", "joint", "collection", "multi_view")
        )
        if single is None:
            raise KeyError(
                "Model output must expose single-view embeddings, for example "
                "output['single']['embedding']"
            )
        batch_size, num_views = images.shape[:2]
        if single.ndim == 2:
            if single.shape[0] != batch_size * num_views:
                raise ValueError(
                    f"Single embeddings have shape {single.shape}; expected first dimension "
                    f"{batch_size * num_views}"
                )
            single = single.reshape(batch_size, num_views, -1)
        elif single.ndim != 3 or single.shape[:2] != (batch_size, num_views):
            raise ValueError(
                f"Single embeddings must be [B*V,D] or [B,V,D], got {single.shape}"
            )
        if joint is not None:
            if joint.ndim == 3 and joint.shape[1] == 1:
                joint = joint[:, 0]
            if joint.ndim != 2 or joint.shape[0] != batch_size:
                raise ValueError(f"Joint embeddings must be [B,D], got {joint.shape}")
        return single, joint

    @torch.inference_mode()
    def extract_gallery(self, dataloader: Iterable, *, cache_path=None,
                        overwrite_cache: bool = False, deduplicate_paths: bool = True):
        """Extract every valid image embedding from a gallery loader."""
        if cache_path is not None and Path(cache_path).is_file() and not overwrite_cache:
            return _load_cache(cache_path, "gallery")
        if self.model is None:
            raise ValueError("A model is required for gallery extraction")
        self.model.eval().to(self.device)
        embedding_chunks, label_chunks, paths = [], [], []
        seen_paths = set()
        num_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
        for batch_index, batch in enumerate(dataloader):
            images, batch_labels, batch_paths, batch_mask = _unpack_batch(batch, "gallery")
            if images.ndim == 4:
                images = images[:, None]
            batch_size, num_views = images.shape[:2]
            mask = _view_mask(batch_mask, batch_size, num_views)
            single, _ = self._forward_embeddings(
                images, mask, return_collection=False
            )
            label_grid = _labels_by_view(batch_labels, batch_size, num_views)
            path_grid = _batch_paths(batch_paths, batch_size, num_views)
            flat_paths = [path_grid[sample][view]
                          for sample in range(batch_size)
                          for view in range(num_views)]
            valid_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten().tolist()
            kept_indices = []
            for index in valid_indices:
                path = flat_paths[index]
                if deduplicate_paths and path is not None and path in seen_paths:
                    continue
                if path is not None:
                    seen_paths.add(path)
                kept_indices.append(index)
                paths.append(path)
            if kept_indices:
                kept_indices = torch.tensor(kept_indices, device=single.device)
                embedding_chunks.append(_as_cpu_float(
                    single.reshape(batch_size * num_views, -1).index_select(
                        0, kept_indices
                    )
                ))
                label_chunks.append(
                    label_grid.reshape(-1).index_select(0, kept_indices.cpu()).clone()
                )
            if (batch_index + 1) % self.log_interval == 0:
                suffix = f"/{num_batches}" if num_batches is not None else ""
                self._log("Gallery embedding batch %d%s", batch_index + 1, suffix)
        if not embedding_chunks:
            raise RuntimeError("Gallery extraction produced no embeddings")
        payload = {
            "version": _CACHE_VERSION,
            "kind": "gallery",
            "embeddings": _normalize(torch.cat(embedding_chunks)).cpu(),
            "labels": torch.cat(label_chunks).long(),
            "paths": paths,
        }
        if cache_path is not None:
            _save_cache(cache_path, payload)
        return payload

    @torch.inference_mode()
    def extract_queries(self, dataloader: Iterable, *, cache_path=None,
                        overwrite_cache: bool = False):
        """Extract joint and per-view embeddings from a collection loader."""
        if cache_path is not None and Path(cache_path).is_file() and not overwrite_cache:
            return _load_cache(cache_path, "query")
        if self.model is None:
            raise ValueError("A model is required for query extraction")
        self.model.eval().to(self.device)
        joint_rows, view_rows, mask_rows, label_rows, path_rows = [], [], [], [], []
        num_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
        for batch_index, batch in enumerate(dataloader):
            images, batch_labels, batch_paths, batch_mask = _unpack_batch(batch, "query")
            if images.ndim == 4:
                images = images[:, None]
            batch_size, num_views = images.shape[:2]
            mask = _view_mask(batch_mask, batch_size, num_views)
            single, joint = self._forward_embeddings(images, mask)
            single = _normalize(single)
            if joint is None:
                # This fallback is useful for embedding-only baselines. The adapted
                # multi-image model is expected to provide a learned joint branch.
                weights = mask.to(single.device, single.dtype)[..., None]
                joint = (single * weights).sum(1) / weights.sum(1).clamp_min(1)
            label_grid = _labels_by_view(batch_labels, batch_size, num_views)
            paths = _batch_paths(batch_paths, batch_size, num_views)
            for sample in range(batch_size):
                joint_rows.append(_as_cpu_float(joint[sample]))
                view_rows.append(_as_cpu_float(single[sample]))
                mask_rows.append(mask[sample].clone().cpu())
                label_rows.append(label_grid[sample, 0].item())
                path_rows.append(paths[sample])
            if (batch_index + 1) % self.log_interval == 0:
                suffix = f"/{num_batches}" if num_batches is not None else ""
                self._log("Query embedding batch %d%s", batch_index + 1, suffix)
        if not joint_rows:
            raise RuntimeError("Query extraction produced no embeddings")
        max_views = max(row.shape[0] for row in view_rows)
        dimension = joint_rows[0].numel()
        padded_views = torch.zeros(len(view_rows), max_views, dimension)
        padded_mask = torch.zeros(len(view_rows), max_views, dtype=torch.bool)
        for index, row in enumerate(view_rows):
            padded_views[index, :len(row)] = row
            padded_mask[index, :len(row)] = mask_rows[index]
        payload = {
            "version": _CACHE_VERSION,
            "kind": "query",
            "joint_embeddings": _normalize(torch.stack(joint_rows)).cpu(),
            "view_embeddings": _normalize(padded_views).cpu(),
            "view_mask": padded_mask,
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "paths": path_rows,
        }
        if cache_path is not None:
            _save_cache(cache_path, payload)
        return payload

    @staticmethod
    def _chunk_top_by_hotel(similarities: torch.Tensor, hotel_indices: torch.Tensor,
                            num_hotels: int, top_m: int) -> torch.Tensor:
        """Exact per-hotel top-m values for one gallery chunk."""
        num_queries, num_images = similarities.shape
        expanded_hotels = hotel_indices[None].expand(num_queries, -1)
        work = similarities.clone() if top_m > 1 else similarities
        ranks = []
        image_indices = torch.arange(num_images, device=similarities.device)[None]
        for rank in range(top_m):
            values = torch.full((num_queries, num_hotels), -torch.inf,
                                device=similarities.device)
            values.scatter_reduce_(1, expanded_hotels, work, reduce="amax",
                                   include_self=True)
            ranks.append(values)
            if rank + 1 == top_m:
                break
            # Remove exactly one winner for each (query, hotel). This preserves
            # duplicate equal-valued images as distinct members of a top-m mean.
            best_for_image = values.gather(1, expanded_hotels)
            candidates = torch.where(
                work == best_for_image,
                image_indices.expand(num_queries, -1),
                torch.full_like(image_indices.expand(num_queries, -1), num_images),
            )
            winners = torch.full((num_queries, num_hotels), num_images,
                                 dtype=torch.long, device=similarities.device)
            winners.scatter_reduce_(1, expanded_hotels, candidates, reduce="amin",
                                    include_self=True)
            selected = image_indices == winners.gather(1, expanded_hotels)
            work.masked_fill_(selected, -torch.inf)
        return torch.stack(ranks, dim=-1)

    def _score_representations(
        self,
        representations: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        gallery_hotel_indices: torch.Tensor,
        num_hotels: int,
        max_top_m: int,
        image_k: int,
        gallery_chunk_size: int,
    ):
        """Return per-hotel top values and optional global image neighbours."""
        representations = representations.to(self.device)
        hotel_values = None
        if max_top_m:
            hotel_values = torch.full(
                (len(representations), num_hotels, max_top_m), -torch.inf,
                device=self.device,
            )
        if image_k:
            image_scores = torch.full((len(representations), image_k), -torch.inf,
                                      device=self.device)
            image_indices = torch.full((len(representations), image_k), -1,
                                       dtype=torch.long, device=self.device)
        else:
            image_scores = image_indices = None

        for gallery_start in range(0, len(gallery_embeddings), gallery_chunk_size):
            gallery_end = min(gallery_start + gallery_chunk_size, len(gallery_embeddings))
            gallery = gallery_embeddings[gallery_start:gallery_end].to(
                self.device, non_blocking=True
            )
            hotel_indices = gallery_hotel_indices[gallery_start:gallery_end].to(
                self.device, non_blocking=True
            )
            similarities = representations @ gallery.T
            if max_top_m:
                local_values = self._chunk_top_by_hotel(
                    similarities, hotel_indices, num_hotels, max_top_m
                )
                hotel_values = torch.cat((hotel_values, local_values), dim=-1).topk(
                    max_top_m, dim=-1
                ).values

            if image_k:
                local_k = min(image_k, similarities.shape[1])
                local_scores, local_indices = similarities.topk(local_k, dim=1)
                local_indices += gallery_start
                candidate_scores = torch.cat((image_scores, local_scores), dim=1)
                candidate_indices = torch.cat((image_indices, local_indices), dim=1)
                image_scores, selected = candidate_scores.topk(image_k, dim=1)
                image_indices = candidate_indices.gather(1, selected)
        return hotel_values, image_indices

    @staticmethod
    def _mean_top_m(top_values: torch.Tensor, top_m: int) -> torch.Tensor:
        selected = top_values[..., :top_m]
        valid = torch.isfinite(selected)
        return selected.masked_fill(~valid, 0).sum(-1) / valid.sum(-1).clamp_min(1)

    @staticmethod
    def _empty_counts(modes, ks):
        return {mode: {k: 0 for k in ks} for mode in modes}

    @staticmethod
    def _format_recalls(counts, denominators):
        return {
            mode: {f"recall@{k}": counts[mode][k] / max(1, denominators[mode])
                   for k in counts[mode]}
            for mode in counts
        }

    @torch.inference_mode()
    def evaluate_embeddings(
        self,
        query_joint: Optional[torch.Tensor],
        query_views: Optional[torch.Tensor],
        query_labels: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        gallery_labels: torch.Tensor,
        *,
        query_view_mask: Optional[torch.Tensor] = None,
        ks: Sequence[int] = (1, 5, 10, 100),
        hotel_top_m: Sequence[int] = (1, 3),
        hybrid_alpha: float = 0.5,
        query_chunk_size: int = 16,
        gallery_chunk_size: int = 32768,
        compute_image_metrics: bool = True,
        compute_hotel_metrics: bool = True,
        cache_gallery_on_device: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate joint, late-interaction, and hybrid hotel retrieval.

        Args:
            query_joint: ``[Q,D]`` learned multi-view collection embeddings.
            query_views: ``[Q,V,D]`` individual embeddings for the same queries.
            query_labels: Hotel identity for each of the ``Q`` collections.
            gallery_embeddings: ``[G,D]`` embeddings of individual gallery images.
            gallery_labels: Hotel identity for each gallery image.
            query_view_mask: Optional ``[Q,V]`` validity mask.
            hotel_top_m: Exact hotel aggregations to evaluate. ``1`` is max;
                values greater than one mean the best available top-m images.
            cache_gallery_on_device: Keep the normalized gallery on the scoring
                device across query chunks. Disable this on memory-limited GPUs.

        ``image`` metrics are the standard image-gallery Recall@K for the learned
        joint descriptor and for every valid constituent view independently.
        """
        if query_joint is None and query_views is None:
            raise ValueError("At least one of query_joint or query_views is required")
        gallery_embeddings = torch.as_tensor(gallery_embeddings, dtype=torch.float32)
        gallery_labels = torch.as_tensor(gallery_labels, dtype=torch.long).flatten().cpu()
        query_labels = torch.as_tensor(query_labels, dtype=torch.long).flatten().cpu()
        if gallery_embeddings.ndim != 2 or len(gallery_embeddings) != len(gallery_labels):
            raise ValueError("Gallery embeddings/labels have incompatible shapes")
        if not len(gallery_embeddings):
            raise ValueError("The gallery is empty")
        if not compute_hotel_metrics and not compute_image_metrics:
            raise ValueError("At least one hotel or image metric must be requested")
        if any(k < 1 for k in ks) or (
            compute_hotel_metrics and any(m < 1 for m in hotel_top_m)
        ):
            raise ValueError("Recall cutoffs and hotel_top_m values must be positive")
        ks = tuple(sorted(set(int(k) for k in ks)))
        hotel_top_m = tuple(sorted(set(int(m) for m in hotel_top_m)))
        if not 0 <= hybrid_alpha <= 1:
            raise ValueError("hybrid_alpha must lie in [0, 1]")

        if query_views is not None:
            query_views = torch.as_tensor(query_views, dtype=torch.float32)
            if query_views.ndim == 2:
                query_views = query_views[:, None]
            if query_views.ndim != 3:
                raise ValueError("query_views must be [Q,V,D] or [Q,D]")
            num_queries, num_views, dimension = query_views.shape
            query_view_mask = _view_mask(query_view_mask, num_queries, num_views)
            if not query_view_mask.any(dim=1).all():
                raise ValueError("Every query collection must have at least one valid view")
        else:
            num_queries, dimension = query_joint.shape
            num_views = 0
            query_view_mask = None
        if query_joint is not None:
            query_joint = torch.as_tensor(query_joint, dtype=torch.float32)
            if query_joint.ndim != 2:
                raise ValueError("query_joint must be [Q,D]")
            if len(query_joint) != num_queries or query_joint.shape[1] != dimension:
                raise ValueError("Joint and view query embeddings have incompatible shapes")
        elif query_views is not None:
            weights = query_view_mask.to(query_views.dtype)[..., None]
            query_joint = (query_views * weights).sum(1) / weights.sum(1)
        if len(query_labels) != num_queries:
            raise ValueError("There must be one query label per query collection")
        if gallery_embeddings.shape[1] != dimension:
            raise ValueError("Query and gallery embedding dimensions differ")

        query_joint = _normalize(query_joint).cpu()
        query_views = None if query_views is None else _normalize(query_views).cpu()
        gallery_embeddings = _normalize(gallery_embeddings).cpu()
        hotel_labels, gallery_hotel_indices = torch.unique(
            gallery_labels, sorted=True, return_inverse=True
        )
        num_hotels = len(hotel_labels)
        if cache_gallery_on_device and self.device.type != "cpu":
            # OpenHotels' complete 384-D float32 gallery is roughly 600 MiB.
            # Keeping it resident avoids retransmitting it for every query
            # chunk; callers on smaller devices can disable this explicitly.
            scoring_gallery = gallery_embeddings.to(self.device)
            scoring_hotel_indices = gallery_hotel_indices.to(self.device)
        else:
            scoring_gallery = gallery_embeddings
            scoring_hotel_indices = gallery_hotel_indices
        device_hotel_labels = (
            hotel_labels.to(self.device) if compute_hotel_metrics else None
        )
        device_gallery_labels = (
            gallery_labels.to(self.device) if compute_image_metrics else None
        )
        max_hotel_k = min(max(ks), num_hotels) if compute_hotel_metrics else 0
        image_k = min(max(ks), len(gallery_embeddings)) if compute_image_metrics else 0

        modes = ["joint"]
        if query_views is not None:
            modes.extend(("views", "hybrid"))
        aggregation_names = ({
            top_m: "max" if top_m == 1 else f"top{top_m}_mean"
            for top_m in hotel_top_m
        } if compute_hotel_metrics else {})
        hotel_counts = {
            name: self._empty_counts(modes, ks)
            for name in aggregation_names.values()
        }
        hotel_denominators = {mode: num_queries for mode in modes}
        image_modes = ["joint"] + (["single_views"] if query_views is not None else [])
        image_counts = self._empty_counts(image_modes, ks)
        image_denominators = {
            "joint": num_queries,
            "single_views": int(query_view_mask.sum()) if query_views is not None else 0,
        }

        for query_start in range(0, num_queries, query_chunk_size):
            query_end = min(query_start + query_chunk_size, num_queries)
            joint = query_joint[query_start:query_end]
            labels = query_labels[query_start:query_end].to(self.device)
            chunk_size = len(joint)
            if query_views is not None:
                views = query_views[query_start:query_end]
                mask = query_view_mask[query_start:query_end]
                valid_views = views[mask]
                view_query_indices = torch.arange(chunk_size)[:, None].expand(
                    -1, num_views
                )[mask].to(self.device)
                representations = torch.cat((joint, valid_views), dim=0)
            else:
                valid_views = None
                view_query_indices = None
                representations = joint

            top_values, image_indices = self._score_representations(
                representations, scoring_gallery, scoring_hotel_indices,
                num_hotels,
                max(hotel_top_m) if compute_hotel_metrics else 0,
                image_k, gallery_chunk_size,
            )
            for top_m, aggregation_name in aggregation_names.items():
                scores = self._mean_top_m(top_values, top_m)
                joint_scores = scores[:chunk_size]
                mode_scores = {"joint": joint_scores}
                if query_views is not None:
                    view_scores = torch.zeros_like(joint_scores)
                    view_scores.index_add_(0, view_query_indices, scores[chunk_size:])
                    view_counts = torch.bincount(
                        view_query_indices, minlength=chunk_size
                    ).to(view_scores.dtype)[:, None]
                    view_scores /= view_counts.clamp_min(1)
                    mode_scores["views"] = view_scores
                    mode_scores["hybrid"] = (
                        hybrid_alpha * joint_scores + (1 - hybrid_alpha) * view_scores
                    )
                for mode, hotel_scores in mode_scores.items():
                    predictions = hotel_scores.topk(max_hotel_k, dim=1).indices
                    predicted_labels = device_hotel_labels[predictions]
                    for k in ks:
                        effective_k = min(k, max_hotel_k)
                        hotel_counts[aggregation_name][mode][k] += int(
                            (predicted_labels[:, :effective_k] == labels[:, None]).any(1).sum()
                        )

            if compute_image_metrics:
                predicted_image_labels = device_gallery_labels[image_indices]
                for k in ks:
                    effective_k = min(k, image_k)
                    image_counts["joint"][k] += int(
                        (predicted_image_labels[:chunk_size, :effective_k] ==
                         labels[:, None]).any(1).sum()
                    )
                    if query_views is not None:
                        view_labels = labels[view_query_indices]
                        image_counts["single_views"][k] += int(
                            (predicted_image_labels[chunk_size:, :effective_k] ==
                             view_labels[:, None]).any(1).sum()
                        )
            self._log("Retrieval query %d/%d (%.1f%%)", query_end, num_queries,
                      100 * query_end / num_queries)

        available = torch.isin(query_labels, hotel_labels)
        result = {
            "num_queries": num_queries,
            "num_gallery_images": len(gallery_embeddings),
            "num_gallery_hotels": num_hotels,
            "num_queries_with_gallery_match": int(available.sum()),
        }
        if compute_hotel_metrics:
            result["hotel"] = {
                name: self._format_recalls(counts, hotel_denominators)
                for name, counts in hotel_counts.items()
            }
        if compute_image_metrics:
            result["image"] = self._format_recalls(image_counts, image_denominators)
        return result

    def evaluate(self, gallery_loader: Iterable, query_loader: Iterable, *,
                 gallery_cache_path=None, query_cache_path=None,
                 overwrite_cache: bool = False, **evaluation_kwargs):
        """Extract (or load cached) embeddings and call ``evaluate_embeddings``."""
        gallery = self.extract_gallery(
            gallery_loader, cache_path=gallery_cache_path,
            overwrite_cache=overwrite_cache,
        )
        queries = self.extract_queries(
            query_loader, cache_path=query_cache_path,
            overwrite_cache=overwrite_cache,
        )
        return self.evaluate_embeddings(
            queries["joint_embeddings"], queries["view_embeddings"], queries["labels"],
            gallery["embeddings"], gallery["labels"],
            query_view_mask=queries["view_mask"], **evaluation_kwargs,
        )
