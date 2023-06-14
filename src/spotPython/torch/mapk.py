import torchmetrics
import torch
import numpy as np


class MAPK(torchmetrics.Metric):
    """
    Mean Average Precision at K (MAPK) metric.

    This class inherits from the `Metric` class of the `torchmetrics` library.

    Args:
        k (int): The number of top predictions to consider when calculating the metric.
        dist_sync_on_step (bool): Whether to synchronize the metric states across processes during the forward pass.

    Attributes:
        total (torch.Tensor): The cumulative sum of the metric scores across all batches.
        count (torch.Tensor): The number of batches processed.

    Example:
        from spotPython.torch.mapk import MAPK
        import torch
        mapk = MAPK(k=2)
        target = torch.tensor([0, 1, 2, 2])
        preds = torch.tensor(
            [
                [0.5, 0.2, 0.2],  # 0 is in top 2
                [0.3, 0.4, 0.2],  # 1 is in top 2
                [0.2, 0.4, 0.3],  # 2 is in top 2
                [0.7, 0.2, 0.1],  # 2 isn't in top 2
            ]
        )
        mapk.update(preds, target)
        print(mapk.compute()) # tensor(0.6250)
    """

    def __init__(self, k=10, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: torch.Tensor, actual: torch.Tensor):
        """
        Update the state variables with a new batch of data.

        Args:
            predicted (torch.Tensor): A 2D tensor containing the predicted scores for each class.
            actual (torch.Tensor): A 1D tensor containing the ground truth labels.


        Raises:
            AssertionError: If `actual` is not a 1D tensor or if `predicted` is not a 2D tensor
            or if `actual` and `predicted` do not have the same number of elements.
        """
        assert len(actual.shape) == 1, "actual must be a 1D tensor"
        assert len(predicted.shape) == 2, "predicted must be a 2D tensor"
        assert actual.shape[0] == predicted.shape[0], "actual and predicted must have the same number of elements"

        # Convert actual to list of lists
        actual = actual.tolist()
        actual = [[a] for a in actual]

        # Convert predicted to list of lists of indices sorted by confidence score
        _, predicted = predicted.topk(k=self.k, dim=1)
        predicted = predicted.tolist()

        score = np.mean([self.apk(p, a, self.k) for p, a in zip(predicted, actual)])
        self.total += score
        self.count += 1

    def compute(self):
        """
        Compute the mean average precision at k.

        Returns:
            float: The mean average precision at k.
        """
        return self.total / self.count

    @staticmethod
    def apk(predicted, actual, k=10):
        """
        Calculate the average precision at k for a single pair of actual and predicted labels.

        Args:
            predicted (list): A list of predicted labels.
            actual (list): A list of ground truth labels.
            k (int): The number of top predictions to consider.

        Returns:
            float: The average precision at k.
        """
        if not actual:
            return 0.0

        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(actual), k)
