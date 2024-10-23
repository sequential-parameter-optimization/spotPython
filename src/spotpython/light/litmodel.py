import os

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import accuracy


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class LitModel(L.LightningModule):
    """
    A LightningModule class for a simple neural network model.

    Attributes:
        l1 (int): The number of neurons in the first hidden layer.
        epochs (int): The number of epochs to train the model for.
        batch_size (int): The batch size to use during training.
        act_fn (str): The activation function to use in the hidden layers.
        optimizer (str): The optimizer to use during training.
        learning_rate (float): The learning rate for the optimizer.
        _L_in (int): The number of input features.
        _L_out (int): The number of output classes.
        model (nn.Sequential): The neural network model.

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> from torchvision.datasets import MNIST
        >>> from torchvision.transforms import ToTensor
        >>> train_data = MNIST(PATH_DATASETS, train=True, download=True, transform=ToTensor())
        >>> train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
        >>> lit_model = LitModel(l1=128, epochs=10, batch_size=BATCH_SIZE, act_fn='relu', optimizer='adam')
        >>> trainer = L.Trainer(max_epochs=10)
        >>> trainer.fit(lit_model, train_loader)
    """

    def __init__(
        self,
        l1: int,
        epochs: int,
        batch_size: int,
        act_fn: str,
        optimizer: str,
        learning_rate: float = 2e-4,
        _L_in: int = 28 * 28,
        _L_out: int = 10,
        *args,
        **kwargs,
    ):
        """
        Initializes the LitModel object.

        Args:
            l1 (int): The number of neurons in the first hidden layer.
            epochs (int): The number of epochs to train the model for.
            batch_size (int): The batch size to use during training.
            act_fn (str): The activation function to use in the hidden layers.
            optimizer (str): The optimizer to use during training.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 2e-4.
            _L_in (int, optional): The number of input features. Defaults to 28 * 28.
            _L_out (int, optional): The number of output classes. Defaults to 10.

        Returns:
           (NoneType): None
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): A tensor containing a batch of input data.

        Returns:
            torch.Tensor: A tensor containing the log probabilities for each class.
        """
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch: tuple) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch: A tuple containing a batch of input data and labels.

        Returns:
            torch.Tensor: A tensor containing the loss for this batch.
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Performs a single validation step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            None
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch: tuple, batch_idx: int) -> tuple:
        """
        Performs a single test step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            tuple: A tuple containing the loss and accuracy for this batch.
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss, acc

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer to use during training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
