import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from spotPython.hyperparameters.optimizer import optimizer_handler


class NetLightRegression(L.LightningModule):
    """
    A LightningModule class for a regresssion neural network model.

    Attributes:
        l1 (int):
            The number of neurons in the first hidden layer.
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
        layers (nn.Sequential):
            The neural network model.

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> from torchvision.datasets import MNIST
        >>> from torchvision.transforms import ToTensor
        >>> train_data = MNIST(PATH_DATASETS,
                               train=True,
                               download=True,
                               transform=ToTensor())
        >>> train_loader = DataLoader(train_data,
                                      batch_size=BATCH_SIZE)
        >>> net_light_base = NetLightRegression(l1=128,
                                          epochs=10,
                                          batch_size=BATCH_SIZE,
                                          initialization='xavier',
                                          act_fn=nn.ReLU(),
                                          optimizer='Adam',
                                          dropout_prob=0.1,
                                          lr_mult=0.1,
                                          patience=5)
        >>> trainer = L.Trainer(max_epochs=10)
        >>> trainer.fit(net_light_base, train_loader)
    """

    def __init__(
        self,
        l1: int,
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
    ):
        """
        Initializes the NetLightRegression object.

        Args:
            l1 (int): The number of neurons in the first hidden layer.
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

        Returns:
            (NoneType): None

        Raises:
            ValueError: If l1 is less than 4.
        Examples:
            >>> from torch.utils.data import DataLoader
            >>> from torchvision.datasets import MNIST
            >>> from torchvision.transforms import ToTensor
            >>> train_data = MNIST(PATH_DATASETS, train=True, download=True, transform=ToTensor())
            >>> train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
            >>> net_light_base = NetLightRegression(l1=128, epochs=10, batch_size=BATCH_SIZE,
                                                initialization='xavier', act_fn=nn.ReLU(),
                                                optimizer='Adam', dropout_prob=0.1, lr_mult=0.1,
                                                patience=5)
            >>> trainer = L.Trainer(max_epochs=10)
            >>> trainer.fit(net_light_base, train_loader)

        """
        super().__init__()
        # Attribute 'act_fn' is an instance of `nn.Module` and is already saved during
        # checkpointing. It is recommended to ignore them
        # using `self.save_hyperparameters(ignore=['act_fn'])`
        # self.save_hyperparameters(ignore=["act_fn"])
        #
        self._L_in = _L_in
        self._L_out = _L_out
        # _L_in and _L_out are not hyperparameters, but are needed to create the network
        self.save_hyperparameters(ignore=["_L_in", "_L_out"])
        # set dummy input array for Tensorboard Graphs
        # set log_graph=True in Trainer to see the graph (in traintest.py)
        self.example_input_array = torch.zeros((batch_size, self._L_in))
        if self.hparams.l1 < 4:
            raise ValueError("l1 must be at least 4")

        hidden_sizes = [self.hparams.l1, self.hparams.l1 // 2, self.hparams.l1 // 2, self.hparams.l1 // 4]

        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [self._L_in] + hidden_sizes
        layer_size_last = layer_sizes[0]
        for layer_size in layer_sizes[1:]:
            layers += [
                nn.Linear(layer_size_last, layer_size),
                self.hparams.act_fn,
                nn.Dropout(self.hparams.dropout_prob),
            ]
            layer_size_last = layer_size
        layers += [nn.Linear(layer_sizes[-1], self._L_out)]
        # nn.Sequential summarizes a list of modules into a single module, applying them in sequence
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): A tensor containing a batch of input data.

        Returns:
            torch.Tensor: A tensor containing the probabilities for each class.
        Examples:
            >>> from torch.utils.data import DataLoader
            >>> from torchvision.datasets import MNIST
            >>> from torchvision.transforms import ToTensor
            >>> train_data = MNIST(PATH_DATASETS, train=True, download=True, transform=ToTensor())
            >>> train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
            >>> net_light_base = NetLightRegression(l1=128,
                                              epochs=10,
                                              batch_size=BATCH_SIZE,
                                              initialization='xavier', act_fn=nn.ReLU(),
                                              optimizer='Adam', dropout_prob=0.1, lr_mult=0.1,
                                              patience=5)

        """
        x = self.layers(x)
        return x

    def training_step(self, batch: tuple) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.

        Returns:
            torch.Tensor: A tensor containing the loss for this batch.
        Examples:
            >>> from torch.utils.data import DataLoader
            >>> from torchvision.datasets import MNIST
            >>> from torchvision.transforms import ToTensor
            >>> train_data = MNIST(PATH_DATASETS, train=True, download=True, transform=ToTensor())
            >>> train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
            >>> net_light_base = NetLightRegression(l1=128,
                                                epochs=10,
                                                batch_size=BATCH_SIZE,
                                                initialization='xavier', act_fn=nn.ReLU(),
                                                optimizer='Adam', dropout_prob=0.1, lr_mult=0.1,
                                                patience=5)
            >>> trainer = L.Trainer(max_epochs=10)
            >>> trainer.fit(net_light_base, train_loader)

        """
        x, y = batch
        y = y.view(len(y), 1)
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        # mae_loss = F.l1_loss(y_hat, y)
        # self.log("train_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_mae_loss", mae_loss, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss

    def validation_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False):
        """
        Performs a single validation step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.
            batch_idx (int): The index of the current batch.
            prog_bar (bool, optional): Whether to display the progress bar. Defaults to False.

        Returns:
            (NoneType): None
        Examples:
            >>> from torch.utils.data import DataLoader
            >>> from torchvision.datasets import MNIST
            >>> from torchvision.transforms import ToTensor
            >>> val_data = MNIST(PATH_DATASETS, train=False, download=True, transform=ToTensor())
            >>> val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
            >>> net_light_base = NetLightRegression(l1=128,
                                                epochs=10,
                                                batch_size=BATCH_SIZE,
                                                initialization='xavier', act_fn=nn.ReLU(),
                                                optimizer='Adam', dropout_prob=0.1, lr_mult=0.1,
                                                patience=5)
            >>> trainer = L.Trainer(max_epochs=10)
            >>> trainer.fit(net_light_base, val_loader)

        """
        x, y = batch
        y = y.view(len(y), 1)
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        # mae_loss = F.l1_loss(y_hat, y)
        # self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=prog_bar)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        return val_loss

    def test_step(self, batch: tuple, batch_idx: int, prog_bar: bool = False) -> tuple:
        """
        Performs a single test step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.
            batch_idx (int): The index of the current batch.
            prog_bar (bool, optional): Whether to display the progress bar. Defaults to False.

        Returns:
            tuple: A tuple containing the loss and accuracy for this batch.
        """
        x, y = batch
        y_hat = self(x)
        y = y.view(len(y), 1)
        val_loss = F.mse_loss(y_hat, y)
        # mae_loss = F.l1_loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer to use during training.

        """
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = optimizer_handler(
            optimizer_name=self.hparams.optimizer, params=self.parameters(), lr_mult=self.hparams.lr_mult
        )
        return optimizer
