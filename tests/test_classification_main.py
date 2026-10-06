"""Classification CLI checks using synthetic metadata/images and a tiny model."""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

import main as classification
from datasets import HotelsDataset
from engine.evaluator import Evaluator


class TinyClassifier(torch.nn.Module):
    def __init__(self, architecture, num_classes, n, pretrained_weights=False, image_size=16):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = 3
        self.n = n
        self.head = torch.nn.Linear(3, num_classes)

    def forward(self, images):
        features = images.mean(dim=(-2, -1))
        logits = self.head(features)
        result = {'single': {'logits': logits.flatten(0, 1)}}
        if self.n > 1:
            result['mv_collection'] = {'logits': self.head(features.mean(dim=1))}
        return result


class ClassificationMainTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        for split in ('gallery', 'test_room', 'test_object'):
            rows = []
            for hotel in ('A', 'B'):
                for index in range(4 if split == 'gallery' else 2):
                    row = {'hotel_id': hotel, 'path': f'{split}/{hotel}/{index}.jpg',
                           'shard': f'{split}-00000.tar', 'room': f'upload-{index // 2}'}
                    rows.append(row)
                    path = self.data / 'images' / Path(row['shard']).stem / row['path']
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (24, 24), color=(index * 20, 40, 60)).save(path)
            (self.data / f'metadata_{split}.json').write_text(json.dumps(rows))
        self.output = self.root / 'output'
        self.logger = logging.getLogger('classification-test')

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, extra=()):
        return classification.parse_args([
            '--data_dir', str(self.data), '--save_dir', str(self.output),
            '--architecture', 'tiny-test', '--num_images', '2', '--image_size', '16',
            '--batch_size', '2', '--eval_batch_size', '2', '--num_workers', '0',
            '--num_epochs', '1', '--use_mutual_distillation_loss', 'false',
            *extra,
        ])

    def checkpoint(self, classes=('A', 'B'), config=None, module_prefix=False):
        model = TinyClassifier('tiny-test', 2, 2)
        state = model.state_dict()
        if module_prefix:
            state = {'module.' + key: value for key, value in state.items()}
        payload = {'model': state}
        if classes is not None:
            payload['classes'] = list(classes)
        if config is not None:
            payload['config'] = config
        path = self.root / 'checkpoint.pth'
        torch.save(payload, path)
        return path

    def test_eval_only_uses_official_splits_without_training_or_optimizer(self):
        checkpoint = self.checkpoint(config=classification._model_config(self.args()))
        args = self.args(['--eval_only', '--checkpoint', str(checkpoint)])
        with mock.patch.object(classification, 'MultiImageHybrid', side_effect=TinyClassifier) as factory, \
                mock.patch.object(classification, 'TrainerEngine') as trainer, \
                mock.patch.object(classification.torch.optim, 'SGD', side_effect=AssertionError('optimizer built')), \
                mock.patch.object(classification, 'OpenHotelsDataset', wraps=classification.OpenHotelsDataset) as datasets, \
                mock.patch.object(torch.cuda, 'is_available', return_value=False):
            scores = classification.main(args, self.logger)
        trainer.assert_not_called()
        self.assertFalse(factory.call_args.kwargs['pretrained_weights'])
        self.assertEqual([call.kwargs['split'] for call in datasets.call_args_list], ['test_room', 'test_object'])
        self.assertEqual(set(scores), {'test_room', 'test_object'})
        self.assertEqual(set(scores['test_room']), {'single', 'mv_collection'})
        self.assertEqual(json.loads((self.output / 'classification_test_metrics.json').read_text()), scores)

    def test_eval_only_accepts_legacy_checkpoint_and_module_prefix(self):
        checkpoint = self.checkpoint(classes=None, module_prefix=True)
        args = self.args(['--eval_only', '--checkpoint', str(checkpoint)])
        with mock.patch.object(classification, 'MultiImageHybrid', side_effect=TinyClassifier), \
                mock.patch.object(torch.cuda, 'is_available', return_value=False):
            scores = classification.main(args, self.logger)
        self.assertEqual(set(scores), {'test_room', 'test_object'})

    def test_eval_only_rejects_wrong_class_order_before_building_model(self):
        checkpoint = self.checkpoint(classes=('B', 'A'))
        args = self.args(['--eval_only', '--checkpoint', str(checkpoint)])
        with mock.patch.object(classification, 'MultiImageHybrid') as factory, \
                self.assertRaisesRegex(ValueError, 'classes/order'):
            classification.main(args, self.logger)
        factory.assert_not_called()

    def test_eval_only_rejects_wrong_image_size_before_building_model(self):
        config = classification._model_config(self.args())
        config['image_size'] = 224
        checkpoint = self.checkpoint(config=config)
        args = self.args(['--eval_only', '--checkpoint', str(checkpoint)])
        with mock.patch.object(classification, 'MultiImageHybrid') as factory, \
                self.assertRaisesRegex(ValueError, 'image_size'):
            classification.main(args, self.logger)
        factory.assert_not_called()

    def test_one_epoch_training_saves_class_config_and_best_metadata(self):
        args = self.args(['--pretrained_weights', 'false'])
        with mock.patch.object(classification, 'MultiImageHybrid', side_effect=TinyClassifier), \
                mock.patch.object(torch.cuda, 'is_available', return_value=False):
            scores = classification.main(args, self.logger)
        payload = classification._read_checkpoint(self.output / 'best.pth')
        self.assertEqual(payload['classes'], ['A', 'B'])
        self.assertEqual(payload['config'], classification._model_config(args))
        self.assertEqual(payload['metadata']['cur_epoch'], 1)
        self.assertEqual(payload['metadata']['best_epoch'], 1)
        self.assertEqual(payload['metadata']['best_score'], payload['metadata']['scores']['mv_collection']['top1_acc'])
        self.assertEqual(set(scores), {'test_room', 'test_object'})

    def test_short_schedules_complete_without_invalid_warmup(self):
        for epochs in range(1, 6):
            with self.subTest(epochs=epochs):
                args = self.args(['--num_epochs', str(epochs)])
                parameter = torch.nn.Parameter(torch.tensor(1.))
                optimizer = torch.optim.SGD([parameter], lr=args.lr)
                scheduler = classification._build_scheduler(optimizer, args, 3)
                for _ in range(epochs * 3):
                    optimizer.step()
                    scheduler.step()
                self.assertEqual(scheduler.last_epoch, scheduler.total_steps)
                self.assertTrue(np.isfinite(optimizer.param_groups[0]['lr']))

    def test_single_image_metrics_deduplicate_paths_within_and_across_batches(self):
        class FixedModel(torch.nn.Module):
            num_classes = 2
            embed_dim = 2
            def forward(self, images):
                return {'single': {'logits': torch.tensor([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])}}
        class Loader:
            def __len__(self): return 2
            def __iter__(self):
                for _ in range(2):
                    yield torch.zeros(2, 2, 3, 4, 4), torch.zeros(2, 2, dtype=torch.long), [('a', 'b'), ('a', 'c')]
        metrics = Evaluator(FixedModel(), n=2, device=torch.device('cpu')).evaluate(Loader())
        self.assertAlmostEqual(metrics['single']['top1_acc'], 2 / 3)

    def test_hotels8k_companions_are_unique_when_enough_sources_exist(self):
        paths = [f'hotel/{index}.jpg' for index in range(4)]
        for path in paths:
            image = self.data / path
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (24, 24)).save(image)
        np.save(self.data / 'train.npy', paths)
        dataset = HotelsDataset(self.data, split='train', n=3, train=True, image_size=16)
        with mock.patch('datasets.hotels8k.np.random.choice', side_effect=[0, 1, 1, 2]):
            _, _, selected = dataset[0]
        self.assertEqual(len(set(selected)), 3)

    def test_eval_only_requires_checkpoint(self):
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            self.args(['--eval_only'])


if __name__ == '__main__':
    unittest.main()
