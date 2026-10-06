import numpy as np
import torch
import torch.nn.functional as F
from utils import to_numpy

def top_k_acc(knn_labels, gt_labels, k):
    accuracy_per_sample = torch.any(knn_labels[:, :k] == gt_labels, dim=1).float()
    return torch.mean(accuracy_per_sample)

class Evaluator(object):


    def __init__(self, model, n=2, device=None, logger=None, log_interval=100):

        self.model = model

        self.extract_device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_device = self.extract_device

        self.num_classes = self.model.num_classes
        self.embed_dim = self.model.embed_dim
        self.n = n
        self.logger = logger
        self.log_interval = log_interval

    @torch.no_grad()
    def extract(self, dataloader):
  
        self.model.eval()
        self.model.to(self.extract_device)

        num_collections = len(dataloader.dataset)
        num_total_images = num_collections * self.n

        results_dict = {'single': {'logits': np.zeros((num_total_images, self.num_classes)),
                                  'classes': np.zeros(num_total_images),
                                  'paths': []},
                       'mv_collection': {'logits': np.zeros((num_collections, self.num_classes)),
                                    'classes': np.zeros(num_collections),
                                    'paths': []}}

        if self.n == 1:
            del results_dict['mv_collection']

        s = 0
        for i, data in enumerate(dataloader):
            images, targets, paths = data
            images = images.to(self.extract_device)
            batch_output = self.model(images)
            e = s + len(images)
            for view_type in batch_output:
                if view_type == 'single':
                    multiplier = self.n
                    t = targets.view(len(images) * self.n)
                    p = np.array(paths).T.flatten().tolist()
                else:
                    multiplier = 1
                    t = targets[:, 0]
                    p = []
                    for w in range(len(paths[0])):
                        l = []
                        for j in range(self.n):
                            l.append(paths[j][w])
                        p.append(l)

                results_dict[view_type]['logits'][s * multiplier: e * multiplier] = to_numpy(batch_output[view_type]['logits'])
                results_dict[view_type]['classes'][s * multiplier: e * multiplier] = to_numpy(t)
                results_dict[view_type]['paths'].extend(p)
            s = e

        results_dict['single']['paths'] = np.array(results_dict['single']['paths'])
        duplicates = set()
        kept_indices = []
        for i in range(len(results_dict['single']['paths'])):
            p = results_dict['single']['paths'][i]
            if p not in duplicates:
                duplicates.add(p)
                kept_indices.append(i)

        for key in results_dict['single']:
            results_dict['single'][key] = results_dict['single'][key][kept_indices]

        for view_type in results_dict:
            results_dict[view_type]['logits'] = torch.from_numpy(results_dict[view_type]['logits']).squeeze()
            results_dict[view_type]['classes'] = results_dict[view_type]['classes'].squeeze()

        return results_dict

    def get_metrics(self, logits, gt_labels):

        try:
            logits = logits
            gt_labels = gt_labels
            predictions = torch.argsort(logits, dim=1, descending=True)
        except torch.cuda.OutOfMemoryError:
            logits = logits.cpu()
            gt_labels = gt_labels.cpu()
            predictions = torch.argsort(logits, dim=1, descending=True)

        class_1 = top_k_acc(predictions, gt_labels, min(1, self.num_classes))
        class_2 = top_k_acc(predictions, gt_labels, min(2, self.num_classes))
        class_5 = top_k_acc(predictions, gt_labels, min(5, self.num_classes))
        class_10 = top_k_acc(predictions, gt_labels, min(10, self.num_classes))
        class_100 = top_k_acc(predictions, gt_labels, min(100, self.num_classes))

        metrics = {
            'top1_acc': class_1.item(),
            'top2_acc': class_2.item(),
            'top5_acc': class_5.item(),
            'top10_acc': class_10.item(),
            'top100_acc': class_100.item(),
        }
        return metrics

    def evaluate(self, dataloader):
        """Compute top-k online instead of allocating examples x classes arrays."""
        self.model.eval()
        self.model.to(self.extract_device)
        ks = (1, 2, 5, 10, 100)
        totals = {}
        seen_single_paths = set()
        num_batches = len(dataloader)
        for batch_index, (images, targets, paths) in enumerate(dataloader):
            images = images.to(self.extract_device, non_blocking=True)
            with torch.autocast(device_type=self.extract_device.type,
                                dtype=torch.bfloat16,
                                enabled=self.extract_device.type == 'cuda'):
                outputs = self.model(images)
            for view_type, output in outputs.items():
                logits = output['logits']
                if view_type == 'single':
                    labels = targets.flatten().to(self.extract_device, non_blocking=True)
                    flat_paths = np.asarray(paths).T.flatten().tolist()
                    keep = []
                    for index, path in enumerate(flat_paths):
                        if path not in seen_single_paths:
                            keep.append(index)
                            seen_single_paths.add(path)
                    if not keep:
                        continue
                    keep = torch.as_tensor(keep, device=logits.device)
                    logits, labels = logits[keep], labels[keep]
                else:
                    labels = targets[:, 0].to(self.extract_device, non_blocking=True)
                max_k = min(max(ks), logits.shape[1])
                predictions = logits.topk(max_k, dim=1).indices
                stats = totals.setdefault(view_type, {'total': 0, **{k: 0 for k in ks}})
                stats['total'] += labels.numel()
                for k in ks:
                    stats[k] += (predictions[:, :min(k, max_k)] == labels[:, None]).any(1).sum().item()
            if (self.logger is not None and
                    ((batch_index + 1) % self.log_interval == 0 or
                     batch_index + 1 == num_batches)):
                self.logger.info('Evaluation batch %d/%d (%.1f%%)',
                                 batch_index + 1, num_batches,
                                 100 * (batch_index + 1) / num_batches)
        return {
            view_type: {f'top{k}_acc': stats[k] / stats['total'] for k in ks}
            for view_type, stats in totals.items()
        }
