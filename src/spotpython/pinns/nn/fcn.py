import torch
import torch.nn as nn


class FCN(nn.Module):
    """A Fully Connected Network (FCN).

    This network consists of an input layer, a specified number of hidden layers,
    and an output layer. All hidden layers use the Tanh activation function.

    Attributes:
        fcs (nn.Sequential): Sequential container for the first linear layer
                             (input to hidden) and its activation.
        fch (nn.Sequential): Sequential container for the hidden layers. Each hidden
                             layer consists of a linear transformation and an activation.
        fce (nn.Linear): The final linear layer (hidden to output).

    References:
        - Solving differential equations using physics informed deep learning: a hand-on tutorial with benchmark tests. Baty, Hubert and Baty, Leo. April 2023.
    """

    def __init__(self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int):
        """Initializes the FCN.

        Args:
            N_INPUT (int): The number of input features.
            N_OUTPUT (int): The number of output features.
            N_HIDDEN (int): The number of neurons in each hidden layer.
            N_LAYERS (int): The total number of layers, including the input layer
                            (which is N_INPUT -> N_HIDDEN), hidden layers, but
                            not counting the final output layer transformation.
                            A N_LAYERS=1 means only input to hidden, then hidden to output.
                            A N_LAYERS=2 means input to hidden, one hidden to hidden, then hidden to output.
                            The number of hidden-to-hidden layers is N_LAYERS - 1.
                            If N_LAYERS is 1, there are no fch layers.

        Examples:
            >>> # Example of creating a FCN
            >>> from spotpython.pinns.nn.fcn import FCN
            >>> model = FCN(N_INPUT=1, N_OUTPUT=1, N_HIDDEN=10, N_LAYERS=3)
            >>> print(model)
            FCN(
              (fcs): Sequential(
                (0): Linear(in_features=1, out_features=10, bias=True)
                (1): Tanh()
              )
              (fch): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=10, out_features=10, bias=True)
                  (1): Tanh()
                )
                (1): Sequential(
                  (0): Linear(in_features=10, out_features=10, bias=True)
                  (1): Tanh()
                )
              )
              (fce): Linear(in_features=10, out_features=1, bias=True)
            )
            >>> # Example of a forward pass
            >>> input_tensor = torch.randn(5, 1) # Batch of 5, 1 input feature
            >>> output_tensor = model(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([5, 1])

            >>> # Example with N_LAYERS = 1 (no hidden-to-hidden layers)
            >>> model_simple = FCN(N_INPUT=2, N_OUTPUT=1, N_HIDDEN=5, N_LAYERS=1)
            >>> print(model_simple)
            FCN(
              (fcs): Sequential(
                (0): Linear(in_features=2, out_features=5, bias=True)
                (1): Tanh()
              )
              (fch): Sequential()
              (fce): Linear(in_features=5, out_features=1, bias=True)
            )
        """
        super().__init__()
        activation = nn.Tanh

        # Input layer: N_INPUT -> N_HIDDEN
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation())

        # Hidden layers: N_HIDDEN -> N_HIDDEN, (N_LAYERS - 1) times
        # If N_LAYERS is 1, range(N_LAYERS-1) is range(0), so fch will be empty.
        hidden_layers = []
        if N_LAYERS > 1:
            for _ in range(N_LAYERS - 1):
                hidden_layers.append(nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation()))
        self.fch = nn.Sequential(*hidden_layers)

        # Output layer: N_HIDDEN -> N_OUTPUT
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor. Shape (batch_size, N_INPUT).

        Returns:
            torch.Tensor: The output tensor. Shape (batch_size, N_OUTPUT).
        """
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
