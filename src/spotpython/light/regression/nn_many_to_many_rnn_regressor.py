import lightning as L
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchmetrics.functional.regression
import torch.optim as optim
from spotpython.hyperparameters.optimizer import optimizer_handler


class ManyToManyRNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size=1,
        rnn_units=256,
        fc_units=256,
        activation_fct=nn.ReLU(),
        dropout=0.0,
        bidirectional=True,
    ):
        super(ManyToManyRNN, self).__init__()
        # Initialize RNN
        self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=rnn_units, num_layers=1, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            rnn_units = rnn_units * 2
        self.fc = nn.Linear(rnn_units, fc_units)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(fc_units, output_size)
        self.activation_fct = activation_fct

    def forward(self, x, lengths):
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn_layer(x)
        x, _ = pad_packed_sequence(packed_output, batch_first=True)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation_fct(x)
        x = self.output_layer(x)
        return x


class ManyToManyRNNRegressor(L.LightningModule):
    def __init__(
        self,
        _L_in: int,
        _L_out: int,
        l1: int = 8,
        rnn_units: int = 256,
        fc_units: int = 256,
        act_fn: nn.Module = nn.ReLU(),
        dropout_prob: float = 0.0,
        bidirectional: bool = True,
        optimizer: str = "Adam",
        lr_mult: float = 1.0,
        patience: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        _torchmetric: str = "mean_squared_error",
        *args,
        **kwargs,
    ):
        super().__init__()
        self._L_in = _L_in
        self._L_out = _L_out
        if _torchmetric is None:
            _torchmetric = "mean_squared_error"
        self._torchmetric = _torchmetric
        self.metric = getattr(torchmetrics.functional.regression, _torchmetric)
        self.save_hyperparameters(ignore=["_L_in", "_L_out", "_torchmetric"])
        self.example_input_array = (torch.zeros((batch_size, 10, _L_in)), torch.tensor([10] * batch_size))

        # Instantiate the RNN layers
        self.layers = ManyToManyRNN(
            input_size=_L_in,
            output_size=_L_out,
            rnn_units=self.hparams.rnn_units,
            fc_units=self.hparams.fc_units,
            activation_fct=self.hparams.act_fn,
            dropout=self.hparams.dropout_prob,
            bidirectional=self.hparams.bidirectional,
        )

    def forward(self, x, lengths) -> torch.Tensor:
        x = self.layers(x, lengths)
        return x

    def _calculate_loss(self, batch):
        x, lengths, y = batch
        y_hat = self(x, lengths)
        y = y.view_as(y_hat)
        loss = self.metric(y_hat, y)
        return loss

    def training_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        val_loss = self._calculate_loss(batch)
        return val_loss

    def validation_step(self, batch: tuple, batch_idx, prog_bar: bool = False) -> torch.Tensor:
        val_loss = self._calculate_loss(batch)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("hp_metric", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        val_loss = self._calculate_loss(batch)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        return val_loss

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizer for the model.

        Notes:
            The default Lightning way is to define an optimizer as
            `optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)`.
            spotpython uses an optimizer handler to create the optimizer, which
            adapts the learning rate according to the lr_mult hyperparameter as
            well as other hyperparameters. See `spotpython.hyperparameters.optimizer.py` for details.

        Returns:
            torch.optim.Optimizer: The optimizer to use during training.

        """
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = optimizer_handler(optimizer_name=self.hparams.optimizer, params=self.parameters(), lr_mult=self.hparams.lr_mult)

        num_milestones = 3  # Number of milestones to divide the epochs
        milestones = [int(self.hparams.epochs / (num_milestones + 1) * (i + 1)) for i in range(num_milestones)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)  # Decay factor

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
