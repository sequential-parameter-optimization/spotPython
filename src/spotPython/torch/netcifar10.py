from torch import nn
import torch.nn.functional as F
from spotPython.utils.file import load_data
import torch.optim as optim
import torch
import os
from torch.utils.data import random_split
import numpy as np
import spotPython.torch.netcore as netcore


class Net_CIFAR10(netcore.Net_Core):
    def __init__(self, l1, l2, lr, batch_size, epochs):
        super(Net_CIFAR10, self).__init__(lr=lr, batch_size=batch_size, epochs=epochs)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)
        #
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        if torch.cuda.device_count() > 1:
            print("We will use", torch.cuda.device_count(), "GPUs!")
        self = nn.DataParallel(self)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
