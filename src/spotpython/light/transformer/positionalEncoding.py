from torch import nn, Tensor
import math
import torch


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
        >>> from spotpython.light.transformer.positionalEncoding import PositionalEncoding
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
