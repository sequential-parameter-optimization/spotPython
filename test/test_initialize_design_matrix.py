import pytest
import numpy as np
from spotpython.spot import Spot
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init

def test_initialize_design_matrix():
    # Initialize function control
    fun_control = fun_control_init(
        tensorboard_log=True,
        TENSORBOARD_CLEAN=True,
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )

    # Define the objective function
    fun = Analytical().fun_sphere

    # Create the Spot instance
    S = Spot(
        fun=fun,
        fun_control=fun_control,
    )

    # Define the starting points
    X_start = np.array([[0.5, 0.5], [0.4, 0.4]])

    # Initialize the design matrix
    S.initialize_design_matrix(X_start)
    design_matrix = S.X

    # Check if the design matrix is not None
    assert design_matrix is not None, "The design matrix should not be None."
    # Check if the design matrix has the expected shape
    assert design_matrix.shape[0] > 0, "The design matrix should have more than 0 rows."
    assert design_matrix.shape[1] == X_start.shape[1], "The design matrix should have the same number of columns as X_start."

    # Check if the starting points are included in the design matrix
    for point in X_start:
        assert any(np.allclose(point, row) for row in design_matrix), f"The point {point} should be included in the design matrix."

if __name__ == "__main__":
    pytest.main()