import tempfile
import unittest
from pathlib import Path

import torch

from engine.retrieval_evaluator import RetrievalEvaluator, exact_cosine_topk


class _EmbeddingModel(torch.nn.Module):
    """Tiny deterministic model implementing the retrieval output contract."""

    def forward(self, images, view_mask=None):
        # The production model's historical/default forward is classification.
        return {"single": {"logits": torch.zeros(images.shape[0], 3)}}

    def forward_embeddings(self, images, view_mask=None):
        embeddings = images[..., 0, 0]
        if view_mask is None:
            view_mask = torch.ones(embeddings.shape[:2], dtype=torch.bool,
                                   device=embeddings.device)
        weights = view_mask.to(embeddings.dtype)[..., None]
        joint = (embeddings * weights).sum(1) / weights.sum(1).clamp_min(1)
        return {
            "single": {"embedding": embeddings.flatten(0, 1)},
            "mv_collection": {"embedding": joint},
        }


class RetrievalEvaluatorTest(unittest.TestCase):

    def test_chunked_cosine_search_matches_dense_search(self):
        generator = torch.Generator().manual_seed(7)
        queries = torch.randn(7, 5, generator=generator)
        gallery = torch.randn(13, 5, generator=generator)
        scores, indices = exact_cosine_topk(
            queries, gallery, 6, device=torch.device("cpu"),
            query_chunk_size=3, gallery_chunk_size=4,
        )
        dense = torch.nn.functional.normalize(queries, dim=-1) @ \
            torch.nn.functional.normalize(gallery, dim=-1).T
        expected_scores, expected_indices = dense.topk(6, dim=1)
        self.assertTrue(torch.equal(indices, expected_indices))
        self.assertTrue(torch.allclose(scores, expected_scores, atol=1e-6))

    def test_exact_hotel_top_m_across_gallery_chunks(self):
        generator = torch.Generator().manual_seed(11)
        queries = torch.nn.functional.normalize(
            torch.randn(3, 4, generator=generator), dim=-1
        )
        gallery = torch.nn.functional.normalize(
            torch.randn(7, 4, generator=generator), dim=-1
        )
        # Include an exact duplicate to exercise one-at-a-time tie removal.
        gallery[1] = gallery[0]
        labels = torch.tensor([7, 7, 7, 9, 9, 11, 11])
        hotels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        evaluator = RetrievalEvaluator(device="cpu", use_amp=False)
        values, _ = evaluator._score_representations(
            queries, gallery, inverse, len(hotels), max_top_m=3,
            image_k=0, gallery_chunk_size=2,
        )

        dense = queries @ gallery.T
        expected = torch.full_like(values, -torch.inf)
        for hotel_index in range(len(hotels)):
            selected = dense[:, labels == hotels[hotel_index]]
            top = selected.topk(min(3, selected.shape[1]), dim=1).values
            expected[:, hotel_index, :top.shape[1]] = top
        self.assertTrue(torch.allclose(values.cpu(), expected, atol=1e-6))

    def test_joint_view_and_hybrid_recall(self):
        gallery = torch.tensor([
            [1.0, 0.0], [0.8, 0.6],
            [0.0, 1.0], [-0.6, 0.8],
            [-1.0, 0.0],
        ])
        gallery_labels = torch.tensor([10, 10, 20, 20, 30])
        joint = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        views = torch.tensor([
            [[1.0, 0.0], [0.8, 0.6]],
            [[0.0, 1.0], [-0.6, 0.8]],
            [[-1.0, 0.0], [-1.0, 0.0]],
        ])
        labels = torch.tensor([10, 20, 30])
        evaluator = RetrievalEvaluator(device="cpu", use_amp=False)
        result = evaluator.evaluate_embeddings(
            joint, views, labels, gallery, gallery_labels,
            ks=(1, 2), hotel_top_m=(1, 2), query_chunk_size=2,
            gallery_chunk_size=2,
        )
        for aggregation in ("max", "top2_mean"):
            for mode in ("joint", "views", "hybrid"):
                self.assertEqual(result["hotel"][aggregation][mode]["recall@1"], 1.0)
        self.assertEqual(result["image"]["joint"]["recall@1"], 1.0)
        self.assertEqual(result["image"]["single_views"]["recall@1"], 1.0)

    def test_image_only_evaluation_skips_hotel_aggregation(self):
        gallery = torch.eye(3)
        labels = torch.tensor([10, 20, 30])
        evaluator = RetrievalEvaluator(device="cpu", use_amp=False)
        result = evaluator.evaluate_embeddings(
            gallery, None, labels, gallery, labels,
            ks=(1,), compute_hotel_metrics=False, compute_image_metrics=True,
            query_chunk_size=2, gallery_chunk_size=2,
        )
        self.assertNotIn("hotel", result)
        self.assertEqual(result["image"]["joint"]["recall@1"], 1.0)

    def test_dict_extraction_deduplication_masks_and_cache(self):
        model = _EmbeddingModel()
        evaluator = RetrievalEvaluator(model=model, device="cpu", use_amp=False)
        # Paths use the view-major structure produced by default_collate.
        gallery_batch = {
            "gallery_images": torch.tensor([
                [[[1.0]], [[0.0]]],
                [[[0.0]], [[1.0]]],
                [[[1.0]], [[1.0]]],
            ]).reshape(3, 1, 2, 1, 1),
            "gallery_labels": torch.tensor([0, 1, 2]),
            "gallery_paths": [("a.jpg", "b.jpg", "c.jpg")],
        }
        query_batch = {
            "query_images": torch.tensor([
                [[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]],
                [[[[0.0]], [[1.0]]], [[[1.0]], [[0.0]]]],
            ]),
            "query_label": torch.tensor([0, 1]),
            "query_mask": torch.tensor([[True, False], [True, True]]),
            # Ragged, batch-major paths are emitted by retrieval_collate_fn.
            "query_paths": [["q0a"], ["q1a", "q1b"]],
        }
        with tempfile.TemporaryDirectory() as directory:
            gallery_cache = Path(directory) / "gallery.pt"
            query_cache = Path(directory) / "queries.pt"
            gallery = evaluator.extract_gallery([gallery_batch], cache_path=gallery_cache)
            queries = evaluator.extract_queries([query_batch], cache_path=query_cache)
            self.assertEqual(gallery["embeddings"].shape, (3, 2))
            self.assertTrue(torch.equal(queries["view_mask"], query_batch["query_mask"]))
            self.assertEqual(queries["paths"], [["q0a", None], ["q1a", "q1b"]])
            self.assertTrue(gallery_cache.is_file())
            self.assertTrue(query_cache.is_file())
            # Empty iterables prove the second calls load rather than re-extract.
            cached_gallery = evaluator.extract_gallery([], cache_path=gallery_cache)
            cached_queries = evaluator.extract_queries([], cache_path=query_cache)
            self.assertTrue(torch.equal(cached_gallery["labels"], gallery["labels"]))
            self.assertTrue(torch.equal(cached_queries["labels"], queries["labels"]))


if __name__ == "__main__":
    unittest.main()
