import lightning as L
import torch
from torch import nn
from spotpython.hyperparameters.optimizer import optimizer_handler
import torchmetrics.functional.regression
import torch.optim as optim
from spotpython.hyperparameters.architecture import get_three_layers


class NNLinearRegressor(L.LightningModule):
    """
    A LightningModule class for a regression neural network model.

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
        batch_norm (bool):
            Whether to use batch normalization or not.
        _L_in (int):
            The number of input features.
        _L_out (int):
            The number of output classes.
        _torchmetric (str):
            The metric to use for the loss function. If `None`,
            then "mean_squared_error" is used.
        layers (nn.Sequential):
            The neural network model.

    Examples:
        >>> from torch.utils.data import DataLoader
            from spotpython.light.regression import NNLinearRegressor
            from torch import nn
            import lightning as L
            import torch
            from torch.utils.data import TensorDataset
            torch.manual_seed(0)
            PATH_DATASETS = './data'
            BATCH_SIZE = 128
            # generate data
            num_samples = 1_000
            input_dim = 10
            X = torch.randn(num_samples, input_dim)  # random data for example
            Y = torch.randn(num_samples, 1)  # random target for example
            data_set = TensorDataset(X, Y)
            train_loader = DataLoader(dataset=data_set, batch_size=BATCH_SIZE)
            test_loader = DataLoader(dataset=data_set, batch_size=BATCH_SIZE)
            val_loader = DataLoader(dataset=data_set, batch_size=BATCH_SIZE)
            batch_x, batch_y = next(iter(train_loader))
            print(batch_x.shape)
            print(batch_y.shape)
            net_light_base = NNLinearRegressor(l1=128,
                                            batch_norm=True,
                                                epochs=10,
                                                batch_size=BATCH_SIZE,
                                                initialization='xavier',
                                                act_fn=nn.ReLU(),
                                                optimizer='Adam',
                                                dropout_prob=0.1,
                                                lr_mult=0.1,
                                                patience=5,
                                                _L_in=input_dim,
                                                _L_out=1,
                                                _torchmetric="mean_squared_error",)
            trainer = L.Trainer(max_epochs=2,  enable_progress_bar=True)
            trainer.fit(net_light_base, train_loader)
            # validation and test should give the same result, because the data is the same
            trainer.validate(net_light_base, val_loader)
            trainer.test(net_light_base, test_loader)
                GPU available: True (mps), used: True
                TPU available: False, using: 0 TPU cores
                HPU available: False, using: 0 HPUs

                | Name   | Type       | Params | Mode  | In sizes  | Out sizes
                ----------------------------------------------------------------------
                0 | layers | Sequential | 20.8 K | train | [128, 10] | [128, 1]
                ----------------------------------------------------------------------
                20.8 K    Trainable params
                0         Non-trainable params
                20.8 K    Total params
                0.083     Total estimated model params size (MB)
                69        Modules in train mode
                0         Modules in eval mode
                torch.Size([128, 10])
                torch.Size([128, 1])
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                ┃      Validate metric      ┃       DataLoader 0        ┃
                ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
                │         hp_metric         │     81.1978988647461      │
                │         val_loss          │     81.1978988647461      │
                └───────────────────────────┴───────────────────────────┘
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                ┃        Test metric        ┃       DataLoader 0        ┃
                ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
                │         hp_metric         │     81.1978988647461      │
                │         val_loss          │     81.1978988647461      │
                └───────────────────────────┴───────────────────────────┘
                [{'val_loss': 81.1978988647461, 'hp_metric': 81.1978988647461}]
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
        batch_norm: bool,
        _L_in: int,
        _L_out: int,
        _torchmetric: str,
        *args,
        **kwargs,
    ):
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
        hidden_sizes = get_three_layers(self._L_in, self.hparams.l1)

        if batch_norm:
            # Add batch normalization layers
            layers = []
            layer_sizes = [self._L_in] + hidden_sizes
            for i in range(len(layer_sizes) - 1):
                current_layer_size = layer_sizes[i]
                next_layer_size = layer_sizes[i + 1]
                layers += [
                    nn.Linear(current_layer_size, next_layer_size),
                    nn.BatchNorm1d(next_layer_size),
                    self.hparams.act_fn,
                    nn.Dropout(self.hparams.dropout_prob),
                ]
            layers += [nn.Linear(layer_sizes[-1], self._L_out)]
        else:
            layers = []
            layer_sizes = [self._L_in] + hidden_sizes
            for i in range(len(layer_sizes) - 1):
                current_layer_size = layer_sizes[i]
                next_layer_size = layer_sizes[i + 1]
                layers += [
                    nn.Linear(current_layer_size, next_layer_size),
                    self.hparams.act_fn,
                    nn.Dropout(self.hparams.dropout_prob),
                ]
            layers += [nn.Linear(layer_sizes[-1], self._L_out)]

        # Wrap the layers into a sequential container
        self.layers = nn.Sequential(*layers)

        # Initialization (Xavier, Kaiming, or Default)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.hparams.initialization == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif self.hparams.initialization == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif self.hparams.initialization == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight)
            elif self.hparams.initialization == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight)
            else:  # "Default"
                nn.init.uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        loss = self._calculate_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

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
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=prog_bar)
        self.log("hp_metric", loss, on_step=False, on_epoch=True, prog_bar=prog_bar)
        return loss

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
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=prog_bar)
        self.log("hp_metric", loss, on_step=False, on_epoch=True, prog_bar=prog_bar)
        return loss

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

        num_milestones = 3  # Number of milestones to divide the epochs
        milestones = [int(self.hparams.epochs / (num_milestones + 1) * (i + 1)) for i in range(num_milestones)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)  # Decay factor

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
