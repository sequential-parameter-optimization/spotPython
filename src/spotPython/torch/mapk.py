import torch
from torchmetrics import Metric
import numpy as np


class MAPK(Metric):
    """Computes the mean average precision at k.
    Args:
        k: Number of predictions to consider
        dist_sync_on_step: Whether to sync the output across all GPUs
        device: Device to use for the computation
    Example:
        >>> from torchmetrics import MAPK
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([[0, 1, 2, 3],
        ...                       [0, 2, 1, 3],
        ...                       [0, 1, 3, 2],
        ...                       [0, 3, 1, 2]])
        >>> mapk = MAPK(k=3)
        >>> mapk(preds, target)
        tensor(0.3333)

        >>> y_pred = torch.tensor([[0.5, 0.2, 0.2],  # 0 is in top 2
                     [0.3, 0.4, 0.2],  # 1 is in top 2
                     [0.2, 0.4, 0.3],  # 2 is in top 2
                     [0.7, 0.2, 0.1]]) # 2 isn't in top 2
        >>> y_true = torch.tensor([0, 1, 2, 2])
        >>> mapk_metric = MAPK(k=2)
        >>> mapk_metric.update(y_pred, y_true)
        >>> result = mapk_metric.compute()
        >>> print(result) # tensor(0.37500)
    """

    def __init__(self, k=3, dist_sync_on_step=False, device=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step, device=device)
        self.k = k
        self.add_state("actual", default=[], dist_reduce_fx="cat")
        self.add_state("predicted", default=[], dist_reduce_fx="cat")

    def update(self, y_pred: torch.Tensor, y: torch.Tensor):
        sorted_prediction_ids = np.argsort(-y_pred.cpu().numpy(), axis=1)
        top_k_prediction_ids = sorted_prediction_ids[:, : self.k]
        self.actual.append(y.cpu().numpy().reshape(-1, 1))
        self.predicted.append(top_k_prediction_ids)

    def compute(self):
        actual = np.concatenate(self.actual)
        predicted = np.concatenate(self.predicted)
        return self.mapk(actual, predicted)

    @staticmethod
    def apk(actual, predicted, k=10):
        if len(predicted) > k:
            predicted = predicted[:k]
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        if not actual:
            return 0.0
        return score / min(len(actual), k)

    def mapk(self, actual, predicted):
        return np.mean([self.apk(a, p, self.k) for a, p in zip(actual, predicted)])
