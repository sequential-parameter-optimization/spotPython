import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """
    Inception block as used in GoogLeNet.

    Notes:
        Description from
        [P. Lippe:INCEPTION, RESNET AND DENSENET](https://lightning.ai/docs/pytorch/stable/)
        An Inception block applies four convolution blocks separately on the same feature map:
        a 1x1, 3x3, and 5x5 convolution, and a max pool operation.
        This allows the network to look at the same data with different receptive fields.
        Of course, learning only 5x5 convolution would be theoretically more powerful.
        However, this is not only more computation and memory heavy but also tends to overfit much easier.
        The 1x1 convolutions are used to reduce the number of input channels to the 3x3 and 5x5 convolutions,
        which reduces the number of parameters and computation.

    Args:
        c_in (int):
            Number of input feature maps from the previous layers
        c_red (dict):
            Dictionary with keys "3x3" and "5x5" specifying
            the output of the dimensionality reducing 1x1 convolutions
        c_out (dict):
            Dictionary with keys "1x1", "3x3", "5x5", and "max"
        act_fn (nn.Module):
            Activation class constructor (e.g. nn.ReLU)


    Examples:
        >>> from spotpython.light.cnn.googlenet import InceptionBlock
            import torch
            import torch.nn as nn
            block = InceptionBlock(3,
                        {"3x3": 32, "5x5": 16},
                        {"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                        nn.ReLU)
            x = torch.randn(1, 3, 32, 32)
            y = block(x)
            y.shape
            torch.Size([1, 64, 32, 32])

    """

    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1), nn.BatchNorm2d(c_out["1x1"]), act_fn()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn(),
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn(),
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn(),
        )

    def forward(self, x) -> torch.Tensor:
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out
