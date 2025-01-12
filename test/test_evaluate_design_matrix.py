import pytest
import numpy as np
from unittest.mock import MagicMock
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.utils.init import fun_control_init


@pytest.fixture
def setup_spot():
    # Configure the necessary function control and objective function
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )
    fun = Analytical().fun_sphere

    # Create and return the Spot instance
    spot_instance = Spot(fun=fun, fun_control=fun_control)
    return spot_instance


def test_evaluate_initial_design(setup_spot):
    # Arrange: set up the Spot instance and initialize the design matrix
    spot_instance = setup_spot
    X0 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    spot_instance.initialize_design_matrix(X_start=X0)

    # Check that the design matrix has been initialized correctly
    assert spot_instance.X.shape[0] > 0, "Design matrix should have non-zero rows after initialization."

    # Act: Evaluate the initial design
    spot_instance.evaluate_initial_design()

    # Assert: Validate the output
    assert spot_instance.X is not None and spot_instance.X.shape[0] > 0, "Evaluated X should have non-zero dimensions."
    assert spot_instance.y is not None and spot_instance.y.shape[0] > 0, "Evaluated y should have non-zero values."
    assert not np.any(np.isnan(spot_instance.y)), "Evaluated y should not contain NaN values."