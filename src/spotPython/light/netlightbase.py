import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import accuracy
from spotPython.torch.mapk import MAPK
from spotPython.hyperparameters.optimizer import optimizer_handler


class NetLightBase(L.LightningModule):
    def __init__(
        self,
        l1,
        epochs,
        batch_size,
        initialization,
        act_fn,
        optimizer,
        dropout_prob,
        lr_mult,
        patience=3,
        _L_in=64,
        _L_out=11,
    ):
        super().__init__()

        # Attribute 'act_fn' is an instance of `nn.Module` and is already saved during checkpointing.
        # It is recommended to ignore them using `self.save_hyperparameters(ignore=['act_fn'])`
        self.save_hyperparameters(ignore=["act_fn"])
        self._L_out = _L_out
        if l1 < 4:
            raise ValueError("l1 must be at least 4")
        self.l1 = l1
        hidden_sizes = [l1, l1 // 2, l1 // 2, l1 // 4]
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.initialization = initialization
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.dropout_prob = dropout_prob
        self.lr_mult = lr_mult
        self.train_mapk = MAPK(k=3)
        self.valid_mapk = MAPK(k=3)
        self.test_mapk = MAPK(k=3)

        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [_L_in] + hidden_sizes
        layer_size_last = layer_sizes[0]
        for layer_size in layer_sizes[1:]:
            layers += [nn.Linear(layer_size_last, layer_size), act_fn, nn.Dropout(self.dropout_prob)]
            layer_size_last = layer_size
        layers += [nn.Linear(layer_sizes[-1], self._L_out)]
        # nn.Sequential summarizes a list of modules into a single module, applying them in sequence
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return F.softmax(x, dim=1)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        # compute cross entropy loss from logits and y
        loss = F.cross_entropy(logits, y)
        # self.train_mapk(logits, y)
        # self.log("train_mapk", self.train_mapk, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, prog_bar=False):
        x, y = batch
        logits = self(x)
        # compute cross entropy loss from logits and y
        loss = F.cross_entropy(logits, y)
        # loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self._L_out)
        self.valid_mapk(logits, y)
        self.log("valid_mapk", self.valid_mapk, on_step=False, on_epoch=True, prog_bar=prog_bar)
        self.log("val_loss", loss, prog_bar=prog_bar)
        self.log("val_acc", acc, prog_bar=prog_bar)
        self.log("hp_metric", loss, prog_bar=prog_bar)

    def test_step(self, batch, batch_idx, prog_bar=False):
        x, y = batch
        logits = self(x)
        # compute cross entropy loss from logits and y
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self._L_out)
        self.test_mapk(logits, y)
        self.log("test_mapk", self.test_mapk, on_step=True, on_epoch=True, prog_bar=prog_bar)
        self.log("val_loss", loss, prog_bar=prog_bar)
        self.log("val_acc", acc, prog_bar=prog_bar)
        self.log("hp_metric", loss, prog_bar=prog_bar)
        return loss, acc

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = optimizer_handler(optimizer_name=self.optimizer, params=self.parameters(), lr_mult=self.lr_mult)
        return optimizer
