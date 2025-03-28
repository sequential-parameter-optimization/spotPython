import lightning as L
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchmetrics.functional.regression
import torch.optim as optim
from spotpython.hyperparameters.optimizer import optimizer_handler


class ManyToManyGRU(nn.Module):
    """A Many-to-Many GRU model for sequence-to-sequence regression tasks.

    This model uses a GRU layer followed by a fully connected layer and an output layer.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features. Defaults to 1.
        gru_units (int): The number of units in the GRU layer. Defaults to 128.
        fc_units (int): The number of units in the fully connected layer. Defaults to 128.
        activation_fct (nn.Module): The activation function to use after the fully connected layer. Defaults to nn.ReLU().
        dropout (float): The dropout probability. Defaults to 0.2.
        bidirectional (bool): Whether the GRU is bidirectional. Defaults to True.
        num_layers (int): The number of GRU layers. Defaults to 2.

    Examples:
        >>> from spotpython.light.regression.nn_many_to_many_gru_regressor import ManyToManyGRU
        >>> import torch
        >>> model = ManyToManyGRU(input_size=10, output_size=1)
        >>> x = torch.randn(16, 10, 10)  # Batch of 16 sequences, each of length 10 with 10 features
        >>> lengths = torch.tensor([10] * 16)  # All sequences have length 10
        >>> output = model(x, lengths)
        >>> print(output.shape)  # Output shape: (16, 10, 1)
    """

    def __init__(
        self,
        input_size,
        output_size=1,
        gru_units=128,
        fc_units=128,
        activation_fct=nn.ReLU(),
        dropout=0.2,
        bidirectional=True,
        num_layers=2,
    ):
        super(ManyToManyGRU, self).__init__()
        self.gru_layer = nn.GRU(
            input_size=input_size,
            hidden_size=gru_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if bidirectional:
            gru_units = gru_units * 2
        self.fc = nn.Linear(gru_units, fc_units)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(fc_units, output_size)
        self.activation_fct = activation_fct

    def forward(self, x, lengths) -> torch.Tensor:
        """Forward pass of the ManyToManyGRU model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            lengths (torch.Tensor): Tensor containing the lengths of each sequence in the batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_size).

        Raises:
            ValueError: If the input tensor is empty or if the lengths tensor is empty.
            RuntimeError: If the lengths tensor does not match the batch size of the input tensor.
        """
        if x.size(0) == 0 or lengths.size(0) == 0:
            raise ValueError("Input tensor or lengths tensor is empty.")
        if x.size(0) != lengths.size(0):
            raise RuntimeError(f"Batch size of input tensor ({x.size(0)}) and lengths tensor ({lengths.size(0)}) must match.")

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru_layer(x)
        x, _ = pad_packed_sequence(packed_output, batch_first=True)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation_fct(x)
        x = self.output_layer(x)
        return x


class ManyToManyGRURegressor(L.LightningModule):
    """A LightningModule for training and evaluating a Many-to-Many GRU regressor.

    Args:
        _L_in (int): The number of input features.
        _L_out (int): The number of output features.
        l1 (int): Unused parameter. Defaults to 8.
        gru_units (int): The number of units in the GRU layer. Defaults to 128.
        fc_units (int): The number of units in the fully connected layer. Defaults to 128.
        act_fn (nn.Module): The activation function to use after the fully connected layer. Defaults to nn.ReLU().
        dropout_prob (float): The dropout probability. Defaults to 0.2.
        bidirectional (bool): Whether the GRU is bidirectional. Defaults to True.
        num_layers (int): The number of GRU layers. Defaults to 2.
        optimizer (str): The optimizer to use. Defaults to "Adam".
        lr_mult (float): Learning rate multiplier. Defaults to 1.0.
        patience (int): Patience for learning rate scheduler. Defaults to 5.
        epochs (int): Number of training epochs. Defaults to 100.
        batch_size (int): Batch size for training. Defaults to 32.
        _torchmetric (str): The metric to use for evaluation. Defaults to "mean_squared_error".

    Examples:
        >>> model = ManyToManyGRURegressor(_L_in=10, _L_out=1)
        >>> x = torch.randn(16, 10, 10)  # Batch of 16 sequences, each of length 10 with 10 features
        >>> lengths = torch.tensor([10] * 16)  # All sequences have length 10
        >>> output = model(x, lengths)
        >>> print(output.shape)  # Output shape: (16, 10, 1)
    """

    def __init__(
        self,
        _L_in: int,
        _L_out: int,
        l1: int = 8,
        gru_units: int = 128,
        fc_units: int = 128,
        act_fn: nn.Module = nn.ReLU(),
        dropout_prob: float = 0.2,
        bidirectional: bool = True,
        num_layers: int = 2,
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

        self.layers = ManyToManyGRU(
            input_size=_L_in,
            output_size=_L_out,
            gru_units=self.hparams.gru_units,
            fc_units=self.hparams.fc_units,
            activation_fct=self.hparams.act_fn,
            dropout=self.hparams.dropout_prob,
            bidirectional=self.hparams.bidirectional,
            num_layers=self.hparams.num_layers,
        )

    def forward(self, x, lengths) -> torch.Tensor:
        """Forward pass of the ManyToManyGRURegressor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            lengths (torch.Tensor): Tensor containing the lengths of each sequence in the batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_size).
        """
        x = self.layers(x, lengths)
        return x

    def _calculate_loss(self, batch):
        """Calculates the loss for a given batch.

        Args:
            batch (tuple): A tuple containing (x, lengths, y), where:
                - x: Input tensor of shape (batch_size, seq_len, input_size).
                - lengths: Tensor containing the lengths of each sequence in the batch.
                - y: Target tensor of shape (batch_size, seq_len, output_size).

        Returns:
            torch.Tensor: The calculated loss.
        """
        x, lengths, y = batch
        y_hat = self(x, lengths)
        y = y.view_as(y_hat)
        loss = self.metric(y_hat, y)
        return loss

    def training_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        """Performs a single training step.

        Args:
            batch (tuple): A tuple containing (x, lengths, y).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The training loss.
        """
        val_loss = self._calculate_loss(batch)
        return val_loss

    def validation_step(self, batch: tuple, batch_idx, prog_bar: bool = False) -> torch.Tensor:
        """Performs a single validation step.

        Args:
            batch (tuple): A tuple containing (x, lengths, y).
            batch_idx (int): The index of the batch.
            prog_bar (bool): Whether to log the loss to the progress bar. Defaults to False.

        Returns:
            torch.Tensor: The validation loss.
        """
        val_loss = self._calculate_loss(batch)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("hp_metric", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        """Performs a single test step.

        Args:
            batch (tuple): A tuple containing (x, lengths, y).
            batch_idx (int): The index of the batch.
            prog_bar (bool): Whether to log the loss to the progress bar. Defaults to False.

        Returns:
            torch.Tensor: The test loss.
        """
        val_loss = self._calculate_loss(batch)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        return val_loss

    def configure_optimizers(self) -> dict:
        """Configures the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler configuration.
        """
        optimizer = optimizer_handler(optimizer_name=self.hparams.optimizer, params=self.parameters(), lr_mult=self.hparams.lr_mult)

        num_milestones = 3
        milestones = [int(self.hparams.epochs / (num_milestones + 1) * (i + 1)) for i in range(num_milestones)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
