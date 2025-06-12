import lightning as L
import torch
from torch import nn
from spotpython.hyperparameters.optimizer import optimizer_handler
import torchmetrics.functional.regression
import torch.optim as optim


class NNFunnelRegressor(L.LightningModule):
    """
    A LightningModule class for a regression neural network model.
    This is a funnel shape neural network with varying number of layers and neurons per layer. An enhanced version of this class is available
    as nn_linear_regression.py in the same directory.

    Attributes:
        l1 (int):
            The number of neurons in the first hidden layer.
        num_layers (int):
            The number of hidden layers in the model.
        epochs (int):
            The number of epochs to train the model for.
        batch_size (int):
            The batch size to use during training.
        initialization (str):
            The initialization method to use for the weights.
        act_fn (nn.Module):
            The activation function to use in the hidden layers.
        optimizer (str):
            The optimizer to use during training.
        dropout_prob (float):
            The probability of dropping out a neuron during training.
        lr_mult (float):
            The learning rate multiplier for the optimizer.
        patience (int):
            The number of epochs to wait before early stopping.
        _L_in (int):
            The number of input features.
        _L_out (int):
            The number of output classes.
        _torchmetric (str):
            The metric to use for the loss function. If `None`,
            then "mean_squared_error" is used.
        layers (nn.Sequential):
            The neural network model.
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
        """
        Initializes the NetLightRegression object.

        Args:
            l1 (int): The number of neurons in the first hidden layer.
            num_layers (int): The number of hidden layers in the model.
            epochs (int): The number of epochs to train the model for.
            batch_size (int): The batch size to use during training.
            initialization (str): The initialization method to use for the weights.
            act_fn (nn.Module): The activation function to use in the hidden layers.
            optimizer (str): The optimizer to use during training.
            dropout_prob (float): The probability of dropping out a neuron during training.
            lr_mult (float): The learning rate multiplier for the optimizer.
            patience (int): The number of epochs to wait before early stopping.
            _L_in (int): The number of input features. Not a hyperparameter, but needed to create the network.
            _L_out (int): The number of output classes. Not a hyperparameter, but needed to create the network.
        _torchmetric (str):
            The metric to use for the loss function. If `None`,
            then "mean_squared_error" is used.

        Returns:
            (NoneType): None

        Raises:
            ValueError: If l1 is less than 8.

        """
        super().__init__()
        # Attribute 'act_fn' is an instance of `nn.Module` and is already saved during
        # checkpointing. It is recommended to ignore them
        # using `self.save_hyperparameters(ignore=['act_fn'])`
        # self.save_hyperparameters(ignore=["act_fn"])
        #
        self._L_in = _L_in
        self._L_out = _L_out
        if _torchmetric is None:
            _torchmetric = "mean_squared_error"
        self._torchmetric = _torchmetric
        self.metric = getattr(torchmetrics.functional.regression, _torchmetric)
        # _L_in and _L_out are not hyperparameters, but are needed to create the network
        # _torchmetric is not a hyperparameter, but is needed to calculate the loss
        self.save_hyperparameters(ignore=["_L_in", "_L_out", "_torchmetric"])
        # set dummy input array for Tensorboard Graphs
        # set log_graph=True in Trainer to see the graph (in traintest.py)
        self.example_input_array = torch.zeros((batch_size, self._L_in))
        if self.hparams.l1 < 8:
            raise ValueError("l1 must be at least 8")

        layers = []
        in_features = self._L_in
        hidden_size = self.hparams.l1
        output_dim = self._L_out

        for i in range(self.hparams.num_layers):
            out_features = max(hidden_size // 2, 8)  # Enforce minimum of 8 units

            layers.append(nn.Linear(in_features, hidden_size))

            if self.hparams.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))  # Add BatchNorm if enabled

            layers.append(self.hparams.act_fn)
            layers.append(nn.Dropout(self.hparams.dropout_prob))

            in_features = hidden_size
            hidden_size = out_features

        layers.append(nn.Linear(in_features, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): A tensor containing a batch of input data.

        Returns:
            torch.Tensor: A tensor containing the output of the model.

        """
        x = self.layers(x)
        return x

    def _calculate_loss(self, batch):
        """
        Calculate the loss for the given batch.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.
            mode (str, optional): The mode of the model. Defaults to "train".

        Returns:
            torch.Tensor: A tensor containing the loss for this batch.

        """
        x, y = batch
        y = y.view(len(y), 1)
        y_hat = self(x)
        loss = self.metric(y_hat, y)
        return loss

    def training_step(self, batch: tuple) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.

        Returns:
            torch.Tensor: A tensor containing the loss for this batch.

        """
        val_loss = self._calculate_loss(batch)
        # self.log("train_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_mae_loss", mae_loss, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss

    def validation_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        """
        Performs a single validation step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.
            batch_idx (int): The index of the current batch.
            prog_bar (bool, optional): Whether to display the progress bar. Defaults to False.

        Returns:
            torch.Tensor: A tensor containing the loss for this batch.

        """
        val_loss = self._calculate_loss(batch)
        # self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=prog_bar)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        return val_loss

    def test_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        """
        Performs a single test step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.
            batch_idx (int): The index of the current batch.
            prog_bar (bool, optional): Whether to display the progress bar. Defaults to False.

        Returns:
            torch.Tensor: A tensor containing the loss for this batch.
        """
        val_loss = self._calculate_loss(batch)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        return val_loss

    def predict_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> torch.Tensor:
        """
        Performs a single prediction step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.
            batch_idx (int): The index of the current batch.
            prog_bar (bool, optional): Whether to display the progress bar. Defaults to False.

        Returns:
            torch.Tensor: A tensor containing the prediction for this batch.
        """
        x, y = batch
        yhat = self(x)
        y = y.view(len(y), 1)
        yhat = yhat.view(len(yhat), 1)
        print(f"Predict step x: {x}")
        print(f"Predict step y: {y}")
        print(f"Predict step y_hat: {yhat}")
        # pred_loss = F.mse_loss(y_hat, y)
        # pred loss not registered
        # self.log("pred_loss", pred_loss, prog_bar=prog_bar)
        # self.log("hp_metric", pred_loss, prog_bar=prog_bar)
        # MisconfigurationException: You are trying to `self.log()`
        # but the loop's result collection is not registered yet.
        # This is most likely because you are trying to log in a `predict` hook, but it doesn't support logging.
        # If you want to manually log, please consider using `self.log_dict({'pred_loss': pred_loss})` instead.
        return (x, y, yhat)

    def configure_optimizers(self) -> torch.optim.Optimizer:
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

        # If the lr_sched hyperparameter is set to True, we will use a learning rate scheduler.
        if self.hparams.lr_sched:
            num_milestones = 3  # Number of milestones to divide the epochs
            milestones = [int(self.hparams.epochs / (num_milestones + 1) * (i + 1)) for i in range(num_milestones)]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)  # Decay factor

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
        #  If the lr_sched hyperparameter is not set to True, we return the optimizer only.
        else:
            return optimizer
