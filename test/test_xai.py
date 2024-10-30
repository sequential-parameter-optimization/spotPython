import numpy as np
import pprint
from spotpython.plot.xai import plot_nn_values_scatter

def test_plot_nn_values_scatter_reshaped_values():
    # Mock data for testing
    nn_values = {
        "Layer 0": np.random.rand(16),  # Expected to fill a 4x4 grid
        "Layer 1": np.random.rand(18),  # Expected to pad and fill a 5x5 grid
    }

    layer_sizes = {
        "Layer 0": np.array([4, 4]),    # Indicates the output for Layer 0 is 4x4
        "Layer 1": np.array([5, 5]),    # Indicates the output for Layer 1 is 5x5
    }

    # Use the modified function that returns reshaped_values for testing,
    # with layer_sizes included
    reshaped_values = plot_nn_values_scatter(nn_values, layer_sizes=layer_sizes, nn_values_names="Test Layer", return_reshaped=True, show=False)

    print("Original nn_values:")
    pprint.pprint(nn_values)
    print("Reshaped nn_values:")
    pprint.pprint(reshaped_values)

    # Assert for Layer 0: Checks if reshaping is correct for perfect square
    assert reshaped_values["Layer 0"].shape == (4, 4)
    # Assert for Layer 1: Checks if reshaping is correct for non-square due to padding
    assert reshaped_values["Layer 1"].shape == (5, 5)