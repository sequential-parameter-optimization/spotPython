from spotPython.light.transformer.skiplinear import SkipLinear
from spotPython.light.transformer.positionalEncoding import PositionalEncoding
from torch import nn
import torch


class TransformerNet(torch.nn.Module):
    def __init__(
        self,
        act_fn,
        _L_in,
        _L_out,
        dropout_prob,
        l_nodes,
        l1,
        dim_feedforward,
        n_head,
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
            l_nodes (int):
                the number of nodes per input
            l1 (int):
                the number of nodes in the first hidden layer
            dim_feedforward (int):
                the hidden size of the feedforward network model
            n_head (int):
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
        self.act_fn = act_fn
        self.dropout_prob = dropout_prob
        self.l_nodes = l_nodes
        self.l1 = l1
        self.n_head = n_head
        self.num_layers = num_layers
        # Each of the _L_1 inputs is forwarded to l_nodes nodes
        self.embed = SkipLinear(_L_in, _L_in * l_nodes)
        self.pos_enc = PositionalEncoding(l_nodes, dropout_prob=dropout_prob)
        # embed_dim "d_model" must be divisible by num_heads
        self.enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=l_nodes, nhead=l_nodes // n_head, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.trans_enc = torch.nn.TransformerEncoder(self.enc_layer, num_layers=num_layers)
        hidden_sizes = [self.l1, self.l1 // 2, self.l1 // 2, self.l1 // 4]
        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [self._L_in * self.l_nodes] + hidden_sizes
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
        z = z.reshape(-1, self._L_in, self.l_nodes)
        z = self.pos_enc(z)
        z = self.trans_enc(z)
        # flatten
        z = z.reshape(-1, self._L_in * self.l_nodes)
        z = self.layers(z)
        return z
