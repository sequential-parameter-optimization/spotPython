# from spotPython.light.transformer.skiplinear import SkipLinear
# from spotPython.light.transformer.positionalEncoding import PositionalEncoding
import torch
from torch import nn, Tensor
import math

# from spotPython.utils.device import getDevice


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
        # device = getDevice()
        self.enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.l_nodes,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
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


class PositionalEncoding(nn.Module):
    """
    Positional encoding module injects some information
    about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed.
    Here, we use ``sine`` and ``cosine`` functions of different frequencies.

    Args:
        d_model (int):
            the embedding dimension. Should be even.
        dropout_prob (float):
            the dropout value
        max_len (int):
            the maximum length of the incoming sequence. Usually related to the max batch_size.
            Can be larger as the batch size, e.g., if prediction is done on a single test set.
            Default: 12552

    Shape:
        Input:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Output:
            Tensor, shape ``[seq_len, batch_size, embedding_dim]``

    Notes:
        * `No return value, but torch`'s method `register_buffer` is used to register the positional encodings.
        * Code adapted from PyTorch: "Transformer Tutorial"


    Reference:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html#positional-encoding

    Examples:
        >>> from spotPython.light.transformer.positionalEncoding import PositionalEncoding
            import torch
            # number of tensors
            n = 3
            # dimension of each tensor, should be even
            k = 10
            pe = PositionalEncoding(d_model=k, dropout_prob=0)
            input = torch.zeros(1, n, k)
            # Generate a tensor of size (1, 10, 4) with values from 1 to 10
            for i in range(n):
                input[0, i, :] = i
            print(f"Input shape: {input.shape}")
            print(f"Input: {input}")
            output = pe(input)
            print(f"Output shape: {output.shape}")
            print(f"Output: {output}")
            position: tensor([[    0],
                            [    1],
                            [    2],
                            ...,
                            [99997],
                            [99998],
                            [99999]])
            div_term: tensor([1.0000e+00, 1.5849e-01, 2.5119e-02, 3.9811e-03, 6.3096e-04])
            Input shape: torch.Size([1, 3, 10])
            Input: tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]]])
            Output shape: torch.Size([1, 3, 10])
            Output: tensor([[[0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                    [1., 2., 1., 2., 1., 2., 1., 2., 1., 2.],
                    [2., 3., 2., 3., 2., 3., 2., 3., 2., 3.]]])
    """

    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 12552) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input tensor.

        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``

        Returns:
            Tensor, shape ``[seq_len, batch_size, embedding_dim]``

        Raises:
            IndexError: if the positional encoding cannot be added to the input tensor
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SkipLinear(torch.nn.Module):
    """
    A skip linear layer.

    Notes:
    Code adapted from James D. McCaffrey:
    "Regression Using a PyTorch Neural Network with a Transformer Component"

    Reference:
        https://jamesmccaffrey.wordpress.com/2023/12/01/regression-using-a-pytorch-neural-network-with-a-transformer-component/

    Args:
        n_in (int):
            the input dimension
        n_out (int):
            the output dimension

    Examples:
        >>> from spotPython.light.transformer.skiplinear import SkipLinear
            import torch
            n_in = 2
            n_out = 4
            sl = SkipLinear(n_in, n_out)
            input = torch.zeros(1, n_in)
            for i in range(n_in):
                input[0, i] = i
            print(f"Input shape: {input.shape}")
            print(f"Input: {input}")
            output = sl(input)
            print(f"Output shape: {output.shape}")
            print(f"Output: {output}")
            print(sl.lst_modules)
            for i in sl.lst_modules:
                print(f"weights: {i.weights}")
            Input shape: torch.Size([1, 2])
            Input: tensor([[0., 1.]])
            Output shape: torch.Size([1, 4])
            Output: tensor([[ 0.0000,  0.0000, -0.0062, -0.0032]], grad_fn=<ViewBackward0>)
            ModuleList(
            (0-1): 2 x Core()
            )
            weights: Parameter containing:
            tensor([[-0.0098],
                    [ 0.0038]], requires_grad=True)
            weights: Parameter containing:
            tensor([[0.0041],
                    [0.0074]], requires_grad=True)
    """

    class Core(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            # 1 node to n nodes, n >= 2
            self.weights = torch.nn.Parameter(torch.zeros((n, 1), dtype=torch.float32))
            self.biases = torch.nn.Parameter(torch.tensor(n, dtype=torch.float32))
            lim = 0.01
            torch.nn.init.uniform_(self.weights, -lim, lim)
            torch.nn.init.zeros_(self.biases)

        def forward(self, x):
            wx = torch.mm(x, self.weights.t())
            v = torch.add(wx, self.biases)
            return v

    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        if n_out % n_in != 0:
            raise ValueError("n_out % n_in != 0")
        n = n_out // n_in  # num nodes per input

        self.lst_modules = torch.nn.ModuleList([SkipLinear.Core(n) for i in range(n_in)])

    def forward(self, x):
        lst_nodes = []
        for i in range(self.n_in):
            xi = x[:, i].reshape(-1, 1)
            oupt = self.lst_modules[i](xi)
            lst_nodes.append(oupt)
        result = torch.cat((lst_nodes[0], lst_nodes[1]), 1)
        for i in range(2, self.n_in):
            result = torch.cat((result, lst_nodes[i]), 1)
        result = result.reshape(-1, self.n_out)
        return result
