from spotPython.light.transformer.skiplinear import SkipLinear
from spotPython.light.transformer.positionalEncoding import PositionalEncoding
from torch import nn
import torch
from spotPython.utils.device import getDevice


class TransformerNet(torch.nn.Module):
    def __init__(
        self,
        act_fn,
        _L_in,
        _L_out,
        dropout_prob,
        d_mult,
        l1,
        dim_feedforward,
        nhead,
        num_layers,
    ):
        """
        A transformer-based regression neural network model.

        Args:
            act_fn (torch.nn.Module):
                the activation function
            _L_in (int):
                the input dimension
            _L_out (int):
                the output dimension
            dropout_prob (float):
                the dropout value
            d_mult (int):
                the multiplier for the number of nodes in the transformer
            l1 (int):
                the number of nodes in the first hidden layer
            dim_feedforward (int):
                the hidden size of the feedforward network model
            nhead (int):
                the number of heads in the multiheadattention models
            num_layers (int):
                the number of sub-encoder-layers in the encoder

        Notes:
            Code adapted from James D. McCaffrey:
            "Regression Using a PyTorch Neural Network with a Transformer Component"

        Reference:
            https://jamesmccaffrey.wordpress.com/2023/12/01/regression-using-a-pytorch-neural-network-with-a-transformer-component/


        """
        super(TransformerNet, self).__init__()
        self._L_in = _L_in
        self._L_out = _L_out
        self.d_mult = d_mult
        self.l1 = l1
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_layers = num_layers
        self.act_fn = act_fn
        self.dropout_prob = dropout_prob
        self.l_nodes = d_mult * nhead * 2
        # Each of the _L_1 inputs is forwarded to d_model nodes,
        # e.g., if _L_in = 90 and d_model = 4, then the input is forwarded to 360 nodes
        # self.embed = SkipLinear(90, 360)
        self.embed = SkipLinear(_L_in, _L_in * self.l_nodes)

        # Positional encoding
        # self.pos_enc = PositionalEncoding(d_model=4, dropout_prob=dropout_prob)
        self.pos_enc = PositionalEncoding(d_model=self.l_nodes, dropout_prob=dropout_prob)

        # Transformer encoder layer
        # embed_dim "d_model" must be divisible by num_heads
        print(f"l_nodes: {self.l_nodes} must be divisible by nhead: {nhead} and 2.")
        # self.enc_layer = torch.nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=10, batch_first=True)
        device = getDevice()
        print(f"device: {device}")
        self.enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.l_nodes, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, device=device
        )

        # Transformer encoder
        # self.trans_enc = torch.nn.TransformerEncoder(self.enc_layer, num_layers=2)
        self.trans_enc = torch.nn.TransformerEncoder(self.enc_layer, num_layers=num_layers)

        # Linear layers (incl. output layer)
        hidden_sizes = [self.l1, self.l1 // 2, self.l1 // 2, self.l1 // 4]
        # Create the network based on the specified hidden sizes
        layers = []

        # layer_sizes = [360] + hidden_sizes
        layer_sizes = [self._L_in * self.l_nodes] + hidden_sizes
        print(f"layer_sizes: {layer_sizes}")
        layer_size_last = layer_sizes[0]
        for layer_size in layer_sizes[1:]:
            layers += [
                nn.Linear(layer_size_last, layer_size),
                nn.BatchNorm1d(layer_size),
                self.act_fn,
                nn.Dropout(self.dropout_prob),
            ]
            layer_size_last = layer_size
        layers += [nn.Linear(layer_sizes[-1], self._L_out)]
        # nn.Sequential summarizes a list of modules into a single module, applying them in sequence
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        z = self.embed(x)

        # z = z.reshape(-1, 90, 4)
        z = z.reshape(-1, self._L_in, self.l_nodes)

        z = self.pos_enc(z)
        z = self.trans_enc(z)

        # flatten
        # z = z.reshape(-1, 360)
        z = z.reshape(-1, self._L_in * self.l_nodes)

        z = self.layers(z)
        return z
