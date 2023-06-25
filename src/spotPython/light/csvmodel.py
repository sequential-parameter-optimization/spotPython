import os

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import accuracy


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class CSVModel(L.LightningModule):
    def __init__(self, l1, epochs, batch_size, act_fn, optimizer, learning_rate=2e-4, _L_in=64, _L_out=11):
        super().__init__()

        # We take in input dimensions as parameters and use those to dynamically build model.
        self._L_out = _L_out
        self.l1 = l1
        self.epochs = epochs
        self.batch_size = batch_size
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_L_in, l1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(l1, l1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(l1, _L_out),
        )

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        # compute cross entropy loss from logits and y
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # compute cross entropy loss from logits and y
        loss = F.cross_entropy(logits, y)
        # loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self._L_out)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # compute cross entropy loss from logits and y
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self._L_out)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
