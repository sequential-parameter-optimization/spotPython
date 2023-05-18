from torch import nn
import spotPython.torch.netcore as netcore


class Net_fashionMNIST(netcore.Net_Core):
    def __init__(self, l1, l2, lr, batch_size, epochs, k_folds, patience, criterion, optimizer):
        super(Net_fashionMNIST, self).__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            k_folds=k_folds,
            patience=patience,
            criterion=criterion,
            optimizer=optimizer,
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, l1), nn.ReLU(), nn.Linear(l1, l2), nn.ReLU(), nn.Linear(l2, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
