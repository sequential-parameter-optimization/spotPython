import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from spotpython.hyperparameters.optimizer import optimizer_handler
import torchmetrics.functional.classification as TMclf
import torch.optim as optim


class NNFunnelClassifier(L.LightningModule):
    """
    Funnel-shaped MLP for classification (binary & multiclass).

    Attributes:
        l1 (int): neurons in first hidden layer.
        num_layers (int): number of hidden layers.
        epochs (int): number of training epochs (used for LR scheduler milestones).
        batch_size (int): batch size (used for example_input_array).
        initialization (str): (keine direkte Nutzung hier – identisch zur Vorlage).
        act_fn (nn.Module): activation module (keine Ignorierung; bleibt tunebar).
        optimizer (str): optimizer name for optimizer_handler.
        dropout_prob (float): dropout probability.
        lr_mult (float): learning-rate multiplier (passed to optimizer_handler).
        patience (int): (nicht in dieser Klasse verwendet – wie Vorlage).
        _L_in (int): input dimension.
        _L_out (int): number of classes. If 1 => binary, else multiclass.
        _torchmetric (str): optional metric name ("accuracy" default). Used for logging, not as loss.
        layers (nn.Sequential): the network.
    """

    def __init__(
        self,
        l1: int,
        num_layers: int,
        epochs: int,
        batch_size: int,
        initialization: str,
        act_fn: nn.Module,
        optimizer: str,
        dropout_prob: float,
        lr_mult: float,
        patience: int,
        _L_in: int,
        _L_out: int,
        _torchmetric: str,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._L_in = _L_in
        self._L_out = _L_out

        # Metric (default accuracy) for logging
        # Loss is always BCEWithLogitsLoss or CrossEntropyLoss
        if _torchmetric is None:
            _torchmetric = "accuracy"
        self._torchmetric = _torchmetric.lower()

        self._is_binary = self._L_out == 1

        self.save_hyperparameters(ignore=["_L_in", "_L_out", "_torchmetric"])

        # Dummy-Input für Graph
        self.example_input_array = torch.zeros((batch_size, self._L_in))

        if self.hparams.l1 < 8:
            raise ValueError("l1 must be at least 8")

        # Netzwerk wie in deiner Vorlage (Funnel, optional BatchNorm/Dropout)
        layers = []
        in_features = self._L_in
        hidden_size = self.hparams.l1
        out_dim = 1 if self._is_binary else self._L_out

        for _ in range(self.hparams.num_layers):
            out_features = max(hidden_size // 2, 8)  # min 8
            layers.append(nn.Linear(in_features, hidden_size))

            if getattr(self.hparams, "batch_norm", False):
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(self.hparams.act_fn)
            layers.append(nn.Dropout(self.hparams.dropout_prob))

            in_features = hidden_size
            hidden_size = out_features

        layers.append(nn.Linear(in_features, out_dim))
        self.layers = nn.Sequential(*layers)

        # Loss nach Task
        if self._is_binary:
            # Combined Sigmoid + BCE
            self._criterion = nn.BCEWithLogitsLoss()
        else:
            # Combined Softmax + CE
            self._criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits. For binary: shape (N,1). For multiclass: (N,C).
        """
        return self.layers(x)

    # internal helper to compute loss and metric
    def _calculate_loss_and_metric(self, batch):
        x, y = batch
        logits = self(x)

        if self._is_binary:
            # y -> (N,1) float
            y_t = y.view(-1, 1).float()
            loss = self._criterion(logits, y_t)
            # Für Metriken bereiten wir Schwellen-Preds vor
            probs = torch.sigmoid(logits).view(-1)
            preds = (probs >= 0.5).long()
            target = y.view(-1).long()
        else:
            # CE expected Long targets (N,) with class indices
            loss = self._criterion(logits, y.long())
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            target = y.long()

        # metrices
        metric_value = None
        try:
            if self._torchmetric == "accuracy":
                if self._is_binary:
                    # binary accuracy (0/1)
                    metric_value = TMclf.accuracy(preds, target, task="binary")
                else:
                    metric_value = TMclf.accuracy(preds, target, task="multiclass", num_classes=self._L_out)
            else:
                # TBC: implement other metrics
                pass
        except Exception:
            metric_value = None

        return loss, metric_value

    # --- Lightning Hooks ---
    def training_step(self, batch: tuple) -> torch.Tensor:
        loss, _ = self._calculate_loss_and_metric(batch)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        val_loss, val_metric = self._calculate_loss_and_metric(batch)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        if val_metric is not None:
            self.log(f"val_{self._torchmetric}", val_metric, prog_bar=prog_bar)
        return val_loss

    def test_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        test_loss, test_metric = self._calculate_loss_and_metric(batch)
        self.log("test_loss", test_loss, prog_bar=prog_bar)
        self.log("hp_metric", test_loss, prog_bar=prog_bar)
        if test_metric is not None:
            self.log(f"test_{self._torchmetric}", test_metric, prog_bar=prog_bar)
        return test_loss

    def predict_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False):
        x, y = batch
        logits = self(x)
        if self._is_binary:
            probs = torch.sigmoid(logits).view(-1, 1)  # (N,1)
            preds = (probs >= 0.5).long()
        else:
            probs = torch.softmax(logits, dim=1)  # (N,C)
            preds = torch.argmax(probs, dim=1, keepdim=True)
        # Debug-Ausgaben wie bei dir:
        print(f"Predict step x: {x}")
        print(f"Predict step y: {y}")
        print(f"Predict step logits: {logits}")
        print(f"Predict step probs: {probs}")
        print(f"Predict step preds: {preds}")
        return (x, y, logits, probs, preds)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = optimizer_handler(optimizer_name=self.hparams.optimizer, params=self.parameters(), lr_mult=self.hparams.lr_mult)

        if getattr(self.hparams, "lr_sched", False):
            num_milestones = 3
            milestones = [int(self.hparams.epochs / (num_milestones + 1) * (i + 1)) for i in range(num_milestones)]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
        else:
            return optimizer
