import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from spotpython.hyperparameters.optimizer import optimizer_handler
import torchmetrics.functional.regression


class RNNLightRegression(L.LightningModule):
    """
    A LightningModule class for a RNN model.

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
        _torchmetric (str):
            The metric to use for the loss function, e.g., "mean_squared_error".
        layers (nn.Sequential):
            The neural network model.

    Examples:
        >>> from torch.utils.data import DataLoader
            from spotpython.data.diabetes import Diabetes
            from spotpython.light.netlightregression import NetLightRegression
            from torch import nn
            import lightning as L
            PATH_DATASETS = './data'
            BATCH_SIZE = 8
            dataset = Diabetes()
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            batch_x, batch_y = next(iter(train_loader))
            print(batch_x.shape)
            print(batch_y.shape)
            net_light_base = NetLightRegression(l1=128,
                                                epochs=10,
                                                batch_size=BATCH_SIZE,
                                                initialization='xavier',
                                                act_fn=nn.ReLU(),
                                                optimizer='Adam',
                                                dropout_prob=0.1,
                                                lr_mult=0.1,
                                                patience=5,
                                                _L_in=10,
                                                _L_out=1,
                                                _torchmetric="mean_squared_error")
            trainer = L.Trainer(max_epochs=2,  enable_progress_bar=True)
            trainer.fit(net_light_base, train_loader)
            trainer.validate(net_light_base, val_loader)
            trainer.test(net_light_base, test_loader)

              | Name   | Type       | Params | In sizes | Out sizes
            -------------------------------------------------------------
            0 | layers | Sequential | 15.9 K | [8, 10]  | [8, 1]
            -------------------------------------------------------------
            15.9 K    Trainable params
            0         Non-trainable params
            15.9 K    Total params
            0.064     Total estimated model params size (MB)

            ─────────────────────────────────────────────────────────────
                Validate metric           DataLoader 0
            ─────────────────────────────────────────────────────────────
                    hp_metric              29010.7734375
                    val_loss               29010.7734375
            ─────────────────────────────────────────────────────────────
            ─────────────────────────────────────────────────────────────
                Test metric             DataLoader 0
            ─────────────────────────────────────────────────────────────
                    hp_metric              29010.7734375
                    val_loss               29010.7734375
            ─────────────────────────────────────────────────────────────

            [{'val_loss': 28981.529296875, 'hp_metric': 28981.529296875}]
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
        _torchmetric: str,
        *args,
        **kwargs,
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
            _torchmetric (str): The metric to use for the loss function, e.g., "mean_squared_error".

        Returns:
            (NoneType): None

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

        # Initialize RNN
        # input_size = number of features is defined via _L_in
        # output size via _L_out
        # num_layers=1: only a single RNN and not stacked
        rnn_units = self.hparams.l1
        fc_units = self.hparams.l1

        # # TODO: make this a hyperparameter
        rnn_nonlinearity = "relu"

        self.rnn_layer = nn.RNN(
            input_size=self._L_in,
            hidden_size=rnn_units,
            num_layers=1,
            nonlinearity=rnn_nonlinearity,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        # # Initialize Hidden- and Output-Layer
        self.fc = nn.Linear(rnn_units, fc_units)
        self.output_layer = nn.Linear(fc_units, self._L_out)

        # # Initialize Activation Function and Dropouts
        # dropout = [0.2, 0, 0]
        # self.dropout1 = nn.Dropout(dropout[0])
        # self.dropout2 = nn.Dropout(dropout[1])
        # self.dropout3 = nn.Dropout(dropout[2])
        # # TODO: use enhanced dropout management for different layers
        self.dropout1 = nn.Dropout(self.hparams.dropout_prob)
        self.dropout2 = nn.Dropout(self.hparams.dropout_prob // 10.0)
        self.dropout3 = nn.Dropout(self.hparams.dropout_prob // 100.0)

        # TODO: Enable different activation functions
        # activation_fct = nn.ReLU()
        # self.activation_fct = activation_fct
        self.activation_fct = self.hparams.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): A tensor containing a batch of input data.

        Returns:
            torch.Tensor: A tensor containing the output of the model.

        """
        # print(f"input: {x.shape}")
        x = self.dropout1(x)
        # print(f"dropout1: {x.shape}")
        x, _ = self.rnn_layer(x)
        # print(f"rnn_layer: {x.shape}")
        # x = x[:, -1, :]
        # print(f"slicing: {x.shape}")
        x = self.dropout2(x)
        # print(f"dropout2: {x.shape}")
        x = self.activation_fct(self.fc(x))
        # print(f"activation_fct: {x.shape}")
        x = self.dropout3(x)
        # print(f"dropout3: {x.shape}")
        x = self.output_layer(x)
        # print(f"output_layer: {x.shape}")
        return x

    def training_step(self, batch: tuple, prog_bar: bool = False) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.
            prog_bar (bool, optional): Whether to display the progress bar. Defaults to False.

        Returns:
            torch.Tensor: A tensor containing the loss for this batch.

        """
        x, y = batch
        # reshape the tensor y to be a column vector (len(y) rows and 1 column)
        y = y.view(len(y), 1)
        # Note: the number of rows in x is equal to the number of rows in y
        y_hat = self(x)
        # Note: the number of rows in y_hat is equal to the number of rows in y
        # train_loss = F.mse_loss(y_hat, y)
        metric = getattr(torchmetrics.functional.regression, self._torchmetric)
        train_loss = metric(y_hat, y)
        # mae_loss = F.l1_loss(y_hat, y)
        # self.log("train_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_mae_loss", mae_loss, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss

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
        x, y = batch
        # reshape the tensor y to be a column vector (len(y) rows and 1 column)
        y = y.view(len(y), 1)
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        # mae_loss = F.l1_loss(y_hat, y)
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
        x, y = batch
        y = y.view(len(y), 1)
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        # mae_loss = F.l1_loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=prog_bar)
        self.log("hp_metric", val_loss, prog_bar=prog_bar)
        return val_loss

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
        optimizer = optimizer_handler(optimizer_name=self.hparams.optimizer, params=self.parameters(), lr_mult=self.hparams.lr_mult)
        return optimizer
