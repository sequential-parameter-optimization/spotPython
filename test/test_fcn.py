import pytest
import torch
import torch.nn as nn
from spotpython.pinns.nn.fcn import FCN

@pytest.mark.parametrize(
    "N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS",
    [
        (1, 1, 10, 1),  # Simplest case: Input -> Hidden -> Output
        (5, 2, 20, 3),  # More complex: Input -> H1 -> H2 -> H3 -> Output
        (10, 1, 5, 2), # Input -> H1 -> H2 -> Output
        (3, 3, 15, 5), # Deeper network
    ]
)
def test_fcn_initialization_architecture(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
    """
    Tests the architecture of the FCN upon initialization.
    Checks layer types, dimensions, and number of hidden layers.
    """
    model = FCN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS)

    # Test fcs (input layer)
    assert isinstance(model.fcs, nn.Sequential), "fcs should be nn.Sequential"
    assert len(model.fcs) == 2, "fcs should have a Linear layer and an activation"
    assert isinstance(model.fcs[0], nn.Linear), "First element of fcs should be nn.Linear"
    assert model.fcs[0].in_features == N_INPUT, f"fcs Linear layer input features should be {N_INPUT}"
    assert model.fcs[0].out_features == N_HIDDEN, f"fcs Linear layer output features should be {N_HIDDEN}"
    assert isinstance(model.fcs[1], nn.Tanh), "Second element of fcs should be nn.Tanh"

    # Test fch (hidden layers)
    assert isinstance(model.fch, nn.Sequential), "fch should be nn.Sequential"
    # The number of hidden-to-hidden layers is N_LAYERS - 1. 
    expected_num_hidden_seqs = N_LAYERS -1 if N_LAYERS > 0 else 0 # N_LAYERS is total layers, fch has N_LAYERS-1 blocks
    if N_LAYERS == 1: # Special case: no hidden-to-hidden layers
        expected_num_hidden_seqs = 0

    assert len(model.fch) == expected_num_hidden_seqs, \
        f"fch should have {expected_num_hidden_seqs} Sequential blocks, got {len(model.fch)}"

    for i in range(expected_num_hidden_seqs):
        hidden_seq = model.fch[i]
        assert isinstance(hidden_seq, nn.Sequential), f"fch block {i} should be nn.Sequential"
        assert len(hidden_seq) == 2, f"fch block {i} should have a Linear layer and an activation"
        assert isinstance(hidden_seq[0], nn.Linear), f"First element of fch block {i} should be nn.Linear"
        assert hidden_seq[0].in_features == N_HIDDEN, f"fch block {i} Linear layer input features should be {N_HIDDEN}"
        assert hidden_seq[0].out_features == N_HIDDEN, f"fch block {i} Linear layer output features should be {N_HIDDEN}"
        assert isinstance(hidden_seq[1], nn.Tanh), f"Second element of fch block {i} should be nn.Tanh"

    # Test fce (output layer)
    assert isinstance(model.fce, nn.Linear), "fce should be nn.Linear"
    assert model.fce.in_features == N_HIDDEN, f"fce Linear layer input features should be {N_HIDDEN}"
    assert model.fce.out_features == N_OUTPUT, f"fce Linear layer output features should be {N_OUTPUT}"


@pytest.mark.parametrize(
    "N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, batch_size",
    [
        (1, 1, 10, 1, 5),
        (5, 2, 20, 3, 10),
        (10, 1, 5, 2, 1), # Single item in batch
        (3, 3, 15, 5, 32),
    ]
)
def test_fcn_forward_pass(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, batch_size):
    """
    Tests the forward pass of the FCN.
    Checks output shape and data type.
    """
    model = FCN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS)
    model.eval() # Set to evaluation mode

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, N_INPUT)

    # Perform forward pass
    output_tensor = model(input_tensor)

    # Check output shape
    expected_shape = (batch_size, N_OUTPUT)
    assert output_tensor.shape == expected_shape, \
        f"Output tensor shape should be {expected_shape}, got {output_tensor.shape}"

    # Check output data type (should be float32 by default for PyTorch nn.Linear)
    assert output_tensor.dtype == torch.float32, \
        f"Output tensor dtype should be torch.float32, got {output_tensor.dtype}"

def test_fcn_n_layers_one():
    """
    Specifically tests the case where N_LAYERS = 1.
    This means fch (hidden-to-hidden layers) should be empty.
    """
    N_INPUT, N_OUTPUT, N_HIDDEN = 2, 1, 5
    N_LAYERS = 1
    model = FCN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS)

    assert isinstance(model.fch, nn.Sequential), "fch should be nn.Sequential"
    assert len(model.fch) == 0, "fch should be empty when N_LAYERS is 1"

    # Test forward pass for this configuration
    batch_size = 3
    input_tensor = torch.randn(batch_size, N_INPUT)
    output_tensor = model(input_tensor)
    expected_shape = (batch_size, N_OUTPUT)
    assert output_tensor.shape == expected_shape, \
        f"Output tensor shape should be {expected_shape}, got {output_tensor.shape}"

# To run these tests, navigate to your project's root directory in the terminal
# and run the command: pytest
# Ensure pytest is installed (pip install pytest) and your spotPython package
# is accessible in the PYTHONPATH.