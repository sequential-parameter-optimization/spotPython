import torch.nn as nn
from spotpython.light.transformer.encoderblock import EncoderBlock


class TransformerEncoder(nn.Module):
    """Transformer encoder module.
    Consists of a stack of EncoderBlocks with layer norm at the end.
    """

    def __init__(self, num_layers, **block_args) -> None:
        """Constructor.
        Args:
            num_layers: int, number of encoder blocks.
            block_args: dict, arguments for EncoderBlock.

        Returns:
            None

        Example:
            >>> from spotpython.light.transformer.encoder import TransformerEncoder
            >>> encoder = TransformerEncoder(num_layers=3,
                                            model_dim=512,
                                            num_heads=8,
                                            dim_feedforward=2048,
                                            dropout=0.1)
            >>> x = torch.rand(10, 32, 512)
            >>> encoder(x).shape
            torch.Size([10, 32, 512])

        """
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps
