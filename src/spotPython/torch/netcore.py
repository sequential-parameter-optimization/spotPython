from torch import nn


class Net_Core(nn.Module):
    def __init__(self, lr, batch_size, epochs, k_folds):
        super(Net_Core, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_folds = k_folds
