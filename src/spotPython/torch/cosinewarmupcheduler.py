import numpy as np
import torch.optim as optim


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warmup learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to use during training.
        warmup (int): The number of warmup steps.
        max_iters (int): The number of maximum iterations the model is trained for.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=100)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
