import torch


class SkipLinear(torch.nn.Module):
    """
    A skip linear layer.

    Notes:
    Code adapted from James D. McCaffrey:
    "Regression Using a PyTorch Neural Network with a Transformer Component"

    Reference:
        https://jamesmccaffrey.wordpress.com/2023/12/01/regression-using-a-pytorch-neural-network-with-a-transformer-component/

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
