import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from spotpython.hyperparameters.optimizer import optimizer_handler
from spotpython.light.transformer.skiplinear import SkipLinear
from spotpython.light.transformer.positionalEncoding import PositionalEncoding
from spotpython.utils.math import generate_div2_list
import torchmetrics.functional.regression


class TransformerLightRegression(L.LightningModule):
    """
    A LightningModule class for a transformer-based regression neural network model.

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
            net_light_base = NetLightRegression2(l1=128,
                                                epochs=10,
                                                batch_size=BATCH_SIZE,
                                                initialization='xavier',
                                                act_fn=nn.ReLU(),
                                                optimizer='Adam',
                                                dropout_prob=0.1,
                                                lr_mult=0.1,
                                                patience=5,
                                                _L_in=10,
                                                _L_out=1)
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
        d_mult: int,
        dim_feedforward: int,
        nhead: int,
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
    ):
        """
        Initializes the TransformerLightRegression object.

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

        Raises:
            ValueError: If l1 is less than 4.

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
        self.d_mult = d_mult
        # set dummy input array for Tensorboard Graphs
        # set log_graph=True in Trainer to see the graph (in traintest.py)
        self.example_input_array = torch.zeros((batch_size, self._L_in))

        # self.l1 = l1
        # self.dim_feedforward = dim_feedforward
        # self.nhead = nhead
        # self.num_layers = num_layers
        # self.act_fn = act_fn
        # self.dropout_prob = dropout_prob

        l_nodes = d_mult * nhead * 2
        # Each of the _L_1 inputs is forwarded to d_model nodes,
        # e.g., if _L_in = 90 and d_model = 4, then the input is forwarded to 360 nodes
        # self.embed = SkipLinear(90, 360)
        self.embed = SkipLinear(_L_in, _L_in * l_nodes)

        # Positional encoding
        # self.pos_enc = PositionalEncoding(d_model=4, dropout_prob=dropout_prob)
        self.pos_enc = PositionalEncoding(d_model=l_nodes, dropout_prob=self.hparams.dropout_prob)

        # Transformer encoder layer
        # embed_dim "d_model" must be divisible by num_heads
        print(f"l_nodes: {l_nodes} must be divisible by nhead: {self.hparams.nhead} and 2.")
        # self.enc_layer = torch.nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=10, batch_first=True)
        self.enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=l_nodes,
            nhead=self.hparams.nhead,
            dim_feedforward=self.hparams.dim_feedforward,
            batch_first=True,
        )

        # Transformer encoder
        # self.trans_enc = torch.nn.TransformerEncoder(self.enc_layer, num_layers=2)
        self.trans_enc = torch.nn.TransformerEncoder(self.enc_layer, num_layers=self.hparams.num_layers)

        n_low = _L_in // 4
        # ensure that n_high is larger than n_low
        n_high = max(self.hparams.l1, 2 * n_low)
        hidden_sizes = generate_div2_list(n_high, n_low)

        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [self._L_in * l_nodes] + hidden_sizes
        layer_size_last = layer_sizes[0]
        for layer_size in layer_sizes[1:]:
            layers += [
                nn.Linear(layer_size_last, layer_size),
                nn.BatchNorm1d(layer_size),
                self.hparams.act_fn,
                nn.Dropout(self.hparams.dropout_prob),
            ]
            layer_size_last = layer_size
        layers += [nn.Linear(layer_sizes[-1], self._L_out)]
        # nn.Sequential summarizes a list of modules into a single module, applying them in sequence
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        l_nodes = self.hparams.d_mult * self.hparams.nhead * 2
        z = self.embed(x)

        # z = z.reshape(-1, 90, 4)
        z = z.reshape(-1, self._L_in, l_nodes)

        z = self.pos_enc(z)
        z = self.trans_enc(z)

        # flatten
        # z = z.reshape(-1, 360)
        z = z.reshape(-1, self._L_in * l_nodes)

        z = self.layers(z)
        return z

    def training_step(self, batch: tuple, prog_bar: bool = False) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (tuple): A tuple containing a batch of input data and labels.

        Returns:
            torch.Tensor: A tensor containing the loss for this batch.

        """
        x, y = batch
        y = y.view(len(y), 1)
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        # mae_loss = F.l1_loss(y_hat, y)
        # self.log("train_loss", val_loss, prog_bar=prog_bar)
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
        x, y = batch
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
        y_hat = self(x)
        y = y.view(len(y), 1)
        val_loss = F.mse_loss(y_hat, y)
        # mae_loss = F.l1_loss(y_hat, y)
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
        optimizer = optimizer_handler(
            optimizer_name=self.hparams.optimizer, params=self.parameters(), lr_mult=self.hparams.lr_mult
        )
        return optimizer
