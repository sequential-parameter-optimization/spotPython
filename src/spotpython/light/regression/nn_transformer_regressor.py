import lightning as L
import torch
from torch import nn
from spotpython.hyperparameters.optimizer import optimizer_handler
import torchmetrics.functional.regression
import torch.optim as optim
from spotpython.light.regression.pos_enc import PositionalEncoding


class NNTransformerRegressor(L.LightningModule):
    def __init__(
        self,
        d_model_mult: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward_mult: int,
        dropout: float,
        epochs: int,
        batch_size: int,
        initialization: str,
        optimizer: str,
        lr_mult: float,
        patience: int,
        _L_in: int,
        _L_out: int,
        _torchmetric: str,
    ):
        super().__init__()

        self._L_in = _L_in
        self._L_out = _L_out

        self.d_model = d_model_mult * nhead
        self.dim_feedforward = dim_feedforward_mult * self.d_model
        print(f"d_model: {self.d_model}, dim_feedforward: {self.dim_feedforward}")

        if _torchmetric is None:
            _torchmetric = "mean_squared_error"
        self._torchmetric = _torchmetric
        self.metric = getattr(torchmetrics.functional.regression, _torchmetric)

        # Embedding layer to convert input features to d_model dimensions
        self.input_proj = nn.Linear(_L_in, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model)

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=nhead, dim_feedforward=self.dim_feedforward, dropout=dropout
            ),
            num_layers=num_encoder_layers,
        )

        # Final regression layer
        self.fc_out = nn.Linear(self.d_model, _L_out)

        # Store hyperparameters
        self.save_hyperparameters(ignore=["_L_in", "_L_out", "_torchmetric"])
        self.example_input_array = torch.zeros((batch_size, _L_in))

    def forward(self, x):
        # Project input features to d_model dimensions
        x = self.input_proj(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer expects input shape: (sequence_length, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Take the output of the last token (assuming the sequence length is fixed)
        x = x.mean(dim=0)

        # Final regression layer
        x = self.fc_out(x)
        return x

    # identical to all nns
    def _calculate_loss(self, batch):
        x, y = batch
        y = y.view(len(y), 1)
        y_hat = self(x)
        loss = self.metric(y_hat, y)
        return loss

    def training_step(self, batch: tuple) -> torch.Tensor:
        val_loss = self._calculate_loss(batch)
        return val_loss

    def validation_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        val_loss = self._calculate_loss(batch)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        return val_loss

    def test_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        val_loss = self._calculate_loss(batch)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        return val_loss

    def predict_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        x, y = batch
        yhat = self(x)
        y = y.view(len(y), 1)
        yhat = yhat.view(len(yhat), 1)
        return (x, y, yhat)

    def configure_optimizers(self):
        optimizer = optimizer_handler(
            optimizer_name=self.hparams.optimizer, params=self.parameters(), lr_mult=self.hparams.lr_mult
        )

        # Dynamic creation of milestones based on the number of epochs.
        num_milestones = 3  # Number of milestones to divide the epochs
        milestones = [int(self.hparams.epochs / (num_milestones + 1) * (i + 1)) for i in range(num_milestones)]

        # Print milestones for debug purposes
        print(f"Milestones: {milestones}")

        # Create MultiStepLR scheduler with dynamic milestones and learning rate multiplier.
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)  # Decay factor

        # Learning rate scheduler configuration
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",  # Adjust learning rate per epoch
            "frequency": 1,  # Apply the scheduler at every epoch
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
