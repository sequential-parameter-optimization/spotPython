import torch


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
        >>> from spotpython.light.transformer.skiplinear import SkipLinear
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
        """A simple linear layer with n outputs."""

        def __init__(self, n):
            """
            Initialize the layer.

            Args:
                n (int): The number of output nodes.
            """
            super().__init__()
            # initialize with random weights using normal distribution
            self.weights = torch.nn.Parameter(torch.randn(1, n))
            # self.weights = torch.nn.Parameter(torch.rand(1, n) * 2 - 1)
            self.linear = torch.nn.Linear(1, n)

        def forward(self, x) -> torch.Tensor:
            """
            Forward pass through the layer.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output of the layer.
            """
            return self.linear(x)

    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        if n_out % n_in != 0:
            raise ValueError("n_out % n_in != 0")
        n = n_out // n_in  # num nodes per input

        self.lst_modules = torch.nn.ModuleList([SkipLinear.Core(n) for _ in range(n_in)])

    def forward(self, x):
        # We want to apply each module to a slice of the input tensor x and collect the outputs.
        # This applies the i-th module to the i-th column of x, reshaped as a column vector.
        # The result is a list of output tensors, which are then concatenated to form the final output.
        lst_nodes = [self.lst_modules[i](x[:, i].unsqueeze(1)) for i in range(self.n_in)]
        result = torch.cat(lst_nodes, dim=1)
        return result.reshape(-1, self.n_out)
