import lightning as L
import torch
import torch.nn.functional as F

# import torch.nn.functional as F
from torch import nn

# from spotpython.hyperparameters.optimizer import optimizer_handler
from spotpython.light.transformer.positionalEncodingBasic import PositionalEncodingBasic
from spotpython.light.transformer.encoder import TransformerEncoder
from spotpython.light.transformer.cosinewarmupscheduler import CosineWarmupScheduler
from torch import optim


class TransformerLightPredictor(L.LightningModule):
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
        model_dim: int,
        num_heads: int,
        lr: float,
        warmup: int,
        max_iters: int,
        input_dropout: float,
        dropout: float,
        *args,
        **kwargs,
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
            _L_in (int):
                The number of input features. Not a hyperparameter, but needed to create the network. `input_dim`,
                hidden dimensionality of the input.
            _L_out (int):
                The number of output classes. Not a hyperparameter, but needed to create the network. `num_classes`,
                number of classes to predict per sequence element.
            model_dim (int):
                Hidden dimensionality to use inside the Transformer
            num_heads (int):
                Number of heads to use in the Multi-Head Attention blocks
            num_layers (int):
                Number of encoder blocks to use.
            lr (float):
                Learning rate in the optimizer
            warmup (int):
                Number of warmup steps. Usually between 50 and 500
            max_iters (int):
                Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            input_dropout (float):
                Dropout to apply on the input features
            dropout (float):
                Dropout to apply inside the Transformer
        """
        super().__init__()
        # Attribute 'act_fn' is an instance of `nn.Module` and is already saved during
        # checkpointing. It is recommended to ignore them
        # using `self.save_hyperparameters(ignore=['act_fn'])`
        # self.save_hyperparameters(ignore=["act_fn"])
        #
        self._L_in = _L_in
        self._L_out = _L_out
        self.d_mult = d_mult
        # _L_in and _L_out are not hyperparameters, but are needed to create the network
        self.save_hyperparameters(ignore=["_L_in", "_L_out"])
        # set dummy input array for Tensorboard Graphs
        # set log_graph=True in Trainer to see the graph (in traintest.py)
        self.example_input_array = torch.zeros((batch_size, self._L_in))
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(nn.Dropout(self.hparams.input_dropout), nn.Linear(self.hparams.input_dim, self.hparams.model_dim))
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncodingBasic(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )
        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class ReversePredictor(TransformerLightPredictor):
    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors
        inp_data, labels = batch
        inp_data = F.one_hot(inp_data, num_classes=self.hparams.num_classes).float()

        # Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, add_positional_encoding=True)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logging
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
