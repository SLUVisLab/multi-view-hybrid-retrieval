import unittest

import torch

from model import MultiImageHybrid


class MultiImageEmbeddingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.model = MultiImageHybrid(
            'vit_tiny_patch16_224', num_classes=7, n=4,
            pretrained_weights=False, image_size=32)
        cls.model.eval()

    def test_default_logits_api_is_preserved(self):
        images = torch.randn(1, 4, 3, 32, 32)
        with torch.no_grad():
            output = self.model(images)
        self.assertEqual(tuple(output['single']['logits'].shape), (4, 7))
        self.assertEqual(tuple(output['mv_collection']['logits'].shape), (1, 7))
        self.assertNotIn('embeddings', output['single'])
        self.assertNotIn('embeddings', output['mv_collection'])

    def test_masked_variable_view_embeddings(self):
        images = torch.randn(3, 4, 3, 32, 32)
        view_mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ])
        with torch.no_grad():
            output = self.model.forward_embeddings(images, view_mask=view_mask)

        single = output['single']['embeddings']
        collection = output['mv_collection']['embeddings']
        self.assertEqual(tuple(single.shape), (12, self.model.embed_dim))
        self.assertEqual(tuple(collection.shape), (3, self.model.embed_dim))
        self.assertTrue(torch.equal(
            output['single']['valid_mask'], view_mask.flatten()))
        self.assertTrue(torch.equal(
            output['mv_collection']['num_views'], torch.tensor([2, 3, 4])))
        self.assertTrue(torch.allclose(
            single.norm(dim=1), torch.ones(12), atol=1e-5))
        self.assertTrue(torch.allclose(
            collection.norm(dim=1), torch.ones(3), atol=1e-5))

        # Padding cannot affect fusion: processing the first collection by
        # itself with only its two real views must produce the same embedding.
        direct = self.model.forward_embeddings(images[:1, :2])
        self.assertTrue(torch.allclose(
            collection[0], direct['mv_collection']['embeddings'][0], atol=1e-5))

    def test_single_dynamic_view_still_has_query_embedding(self):
        images = torch.randn(2, 1, 3, 32, 32)
        with torch.no_grad():
            output = self.model(
                images, return_embeddings=True, return_logits=False)
        self.assertEqual(tuple(output['single']['embeddings'].shape),
                         (2, self.model.embed_dim))
        self.assertEqual(tuple(output['mv_collection']['embeddings'].shape),
                         (2, self.model.embed_dim))

    def test_gallery_can_skip_collection_branch(self):
        images = torch.randn(2, 1, 3, 32, 32)
        with torch.no_grad():
            output = self.model.forward_embeddings(
                images, return_collection=False)
        self.assertEqual(set(output), {'single'})
        self.assertEqual(tuple(output['single']['embeddings'].shape),
                         (2, self.model.embed_dim))

    def test_state_dict_strict_round_trip(self):
        clone = MultiImageHybrid(
            'vit_tiny_patch16_224', num_classes=7, n=4,
            pretrained_weights=False, image_size=32)
        clone.load_state_dict(self.model.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
