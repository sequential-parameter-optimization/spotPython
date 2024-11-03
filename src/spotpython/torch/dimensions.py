import numpy as np
import torch.nn as nn


def extract_linear_dims(model) -> np.array:
    """Extracts the input and output dimensions of the Linear layers in a PyTorch model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        np.array: Array with the input and output dimensions of the Linear layers.

    Examples:
        >>> from spotpython.torch.dimensions import extract_linear_dims
        >>> net = NNLinearRegressor()
        >>> result = extract_linear_dims(net)

    """
    dims = []
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            # Append input and output features of the Linear layer
            dims.append(layer.in_features)
            dims.append(layer.out_features)
    return np.array(dims)
