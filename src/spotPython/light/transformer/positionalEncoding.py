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
            the embedding dimension
        dropout_prob (float):
            the dropout value
        max_len (int):
            the maximum length of the incoming sequence

    Shape:
        Input:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Output:
            Tensor, shape ``[seq_len, batch_size, embedding_dim]``

    Reference:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html#positional-encoding

    Examples:
        >>> from spotPython.light.transformer.positionalEncoding import PositionalEncoding
        >>> import torch
        >>> pe = PositionalEncoding(d_model=20, dropout_prob=0)
        >>> input = torch.zeros(1, 10, 20)
        >>> output = pe(input)
        >>> print(output.shape)
        torch.Size([1, 10, 20])



    """

    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 12552):
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
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
