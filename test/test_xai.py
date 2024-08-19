import numpy as np
import pprint
from spotpython.plot.xai import plot_nn_values_scatter


def test_plot_nn_values_scatter_reshaped_values():
    # Mock data for testing
    nn_values = {
        "layer1": np.random.rand(16),  # 16 values suggesting a perfect square (4x4)
        "layer2": np.random.rand(18),  # 18 values suggesting padding will be required for a 5x5 shape
    }

    # Use the modified function that returns reshaped_values for testing
    reshaped_values = plot_nn_values_scatter(nn_values, "Test Layer1", return_reshaped=True, show=False)

    pprint.pprint(nn_values)
    pprint.pprint(reshaped_values)
    # Assert for layer1: Checks if reshaping is correct for perfect square
    assert reshaped_values["layer1"].shape == (4, 4)
    # Assert for layer2: Checks if reshaping is correct for non-square
    assert reshaped_values["layer2"].shape == (5, 5)
