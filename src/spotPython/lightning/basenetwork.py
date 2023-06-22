from torch import nn


class BaseNetwork(nn.Module):
    def __init__(self, act_fn, _L_in=784, _L_out=10, l1=512, optimizer="Adam", batch_size=16, epochs=2):
        """
        Args:
            act_fn: Object of the activation function that should be used as non-linearity in the network.
            _L_in: input_size, e.g., size of the input images in pixels
            _L_out: e.g., num_classes, number of classes we want to predict
            l1: hidden layer sizes in the NN
        """
        super().__init__()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

        # Create the network based on the specified hidden sizes
        layers = []
        hidden_sizes = [512, 256, 256, 128]
        layer_sizes = [_L_in] + hidden_sizes
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]), act_fn]
        layers += [nn.Linear(layer_sizes[-1], _L_out)]
        # A module list registers a list of modules as submodules (e.g. for parameters)
        self.layers = nn.ModuleList(layers)

        # self.config = {
        #     "act_fn": act_fn.__class__.__name__,
        #     "_L_in": _L_in,
        #     "_L_out": _L_out,
        #     "l1": l1,
        #     "optimizer": optimizer,
        # }

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x
