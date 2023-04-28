from torch import nn
import torch.nn.functional as F
from spotPython.utils.file import load_data
import torch.optim as optim
import torch
import os
from torch.utils.data import random_split
import numpy as np
import spotPython.torch.netcore as netcore


class Net_fashionMNIST(netcore.Net_Core):
    def __init__(self, l1, l2, lr, batch_size, epochs):
        super(Net_fashionMNIST, self).__init__(lr=lr, batch_size=batch_size, epochs=epochs)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, l1), nn.ReLU(), nn.Linear(l1, l2), nn.ReLU(), nn.Linear(l2, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
