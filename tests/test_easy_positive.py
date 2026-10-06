import math
import unittest

import torch

from loss import EPAllLoss, EPHNLoss, EPSHNLoss


def unit_vector_with_similarity(similarity):
    return torch.tensor(
        [similarity, math.sqrt(1.0 - similarity ** 2)], dtype=torch.float32)


class EasyPositiveLossTest(unittest.TestCase):

    def test_negative_mining_variants(self):
        query = torch.tensor([[1.0, 0.0]], requires_grad=True)
        gallery = torch.stack((
            unit_vector_with_similarity(0.8),   # easy positive
            unit_vector_with_similarity(0.2),   # other positive
            unit_vector_with_similarity(0.9),   # hard negative
            unit_vector_with_similarity(0.5),   # semi-hard negative
        )).requires_grad_()
        query_labels = torch.tensor([0])
        gallery_labels = torch.tensor([0, 0, 1, 2])

        ephn, hard_details = EPHNLoss(temperature=1.0)(
            query, gallery, query_labels, gallery_labels,
            return_details=True)
        epshn, semi_details = EPSHNLoss(temperature=1.0)(
            query, gallery, query_labels, gallery_labels,
            return_details=True)

        self.assertAlmostEqual(
            hard_details['positive_similarity'].item(), 0.8, places=5)
        self.assertAlmostEqual(
            hard_details['negative_similarity'].item(), 0.9, places=5)
        self.assertAlmostEqual(
            semi_details['negative_similarity'].item(), 0.5, places=5)
        self.assertAlmostEqual(
            ephn.item(), torch.nn.functional.softplus(torch.tensor(0.1)).item(),
            places=5)
        self.assertAlmostEqual(
            epshn.item(), torch.nn.functional.softplus(torch.tensor(-0.3)).item(),
            places=5)

        (ephn + epshn).backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(gallery.grad)
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_ep_all_matches_nca_formula(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        loss = EPAllLoss(temperature=1.0)(
            query, gallery, torch.tensor([0]), torch.tensor([0, 1, 2]))
        expected = torch.logsumexp(torch.tensor([1.0, 0.0, -1.0]), dim=0) - 1.0
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_same_tensor_excludes_diagonal(self):
        embeddings = torch.tensor([
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [-0.6, 0.8],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        _, details = EPHNLoss(temperature=1.0)(
            embeddings, embeddings, labels, labels, return_details=True)
        self.assertTrue(torch.equal(
            details['positive_index'], torch.tensor([1, 0, 3, 2])))
        self.assertTrue(details['valid_query_mask'].all())

    def test_missing_positive_returns_differentiable_zero(self):
        query = torch.tensor([[1.0, 0.0]], requires_grad=True)
        gallery = torch.tensor([[0.0, 1.0]], requires_grad=True)
        loss, details = EPSHNLoss()(query, gallery, torch.tensor([0]),
                                     torch.tensor([1]), return_details=True)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(details['num_valid'].item(), 0)
        loss.backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(gallery.grad)

    def test_positive_mask_can_exclude_source_image(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
        positive_mask = torch.tensor([[False, True, True]])
        _, details = EPHNLoss(temperature=1.0)(
            query, gallery, torch.tensor([0]), torch.tensor([0, 0, 1]),
            positive_mask=positive_mask, return_details=True)
        self.assertEqual(details['positive_index'].item(), 1)


    def _check_autocast_training(self, device):
        query_values = torch.tensor([
            [1.0, 0.1, 0.2],
            [0.2, 1.0, 0.1],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], device=device)
        gallery_values = torch.tensor([
            [1.0, 0.2, 0.1],
            [0.8, -0.3, 0.2],
            [0.1, 1.0, 0.2],
            [-0.2, 0.8, 0.3],
        ], device=device)
        query_labels = torch.tensor([0, 1, -1, 2], device=device)
        gallery_labels = torch.tensor([0, 0, 1, 1], device=device)
        for loss_type in (EPAllLoss, EPHNLoss, EPSHNLoss):
            with self.subTest(device=device, mode=loss_type.__name__):
                criterion = loss_type(reduction='none')
                reference_query = query_values.clone().requires_grad_()
                reference_gallery = gallery_values.clone().requires_grad_()
                expected = criterion(
                    reference_query, reference_gallery,
                    query_labels, gallery_labels)
                expected.sum().backward()

                query = query_values.clone().requires_grad_()
                gallery = gallery_values.clone().requires_grad_()
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    losses, details = criterion(
                        query, gallery, query_labels, gallery_labels,
                        return_details=True)
                self.assertEqual(losses.dtype, torch.float32)
                self.assertEqual(details['loss_per_query'].dtype, torch.float32)
                self.assertTrue(torch.isfinite(losses).all())
                self.assertTrue(torch.equal(
                    details['valid_query_mask'],
                    torch.tensor([True, True, False, False], device=device)))
                self.assertTrue(torch.equal(
                    losses[2:], torch.zeros(2, device=device)))
                torch.testing.assert_close(losses, expected, rtol=1e-6, atol=1e-7)

                losses.sum().backward()
                self.assertTrue(torch.isfinite(query.grad).all())
                self.assertTrue(torch.isfinite(gallery.grad).all())
                torch.testing.assert_close(
                    query.grad, reference_query.grad, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(
                    gallery.grad, reference_gallery.grad, rtol=1e-6, atol=1e-7)

    def test_cpu_bfloat16_autocast_matches_float32_training(self):
        self._check_autocast_training('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_bfloat16_autocast_matches_float32_training(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest('CUDA bfloat16 is unsupported')
        self._check_autocast_training('cuda')

    def test_bfloat16_embeddings_preserve_self_exclusion_and_gradients(self):
        values = torch.tensor([
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [-0.6, 0.8],
        ], dtype=torch.bfloat16)
        labels = torch.tensor([0, 0, 1, 1])
        for loss_type in (EPAllLoss, EPHNLoss, EPSHNLoss):
            with self.subTest(mode=loss_type.__name__):
                embeddings = values.clone().requires_grad_()
                loss, details = loss_type()(
                    embeddings, embeddings, labels, labels,
                    return_details=True)
                reference_embeddings = values.float().requires_grad_()
                expected = loss_type()(
                    reference_embeddings, reference_embeddings, labels, labels)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.equal(
                    details['positive_index'], torch.tensor([1, 0, 3, 2])))
                self.assertTrue(details['valid_query_mask'].all())
                torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
                loss.backward()
                self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_float64_embeddings_retain_precision(self):
        query = torch.tensor([[1.0, 0.1]], dtype=torch.float64, requires_grad=True)
        gallery = torch.tensor(
            [[0.8, 0.6], [0.0, 1.0]], dtype=torch.float64, requires_grad=True)
        loss = EPSHNLoss()(query, gallery, torch.tensor([0]), torch.tensor([0, 1]))
        self.assertEqual(loss.dtype, torch.float64)
        loss.backward()
        self.assertEqual(query.grad.dtype, torch.float64)
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(gallery.grad).all())



if __name__ == '__main__':
    unittest.main()
