import pytest
import numpy as np
from spotpython.spot import spot
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, design_control_init

def test_run_method():
    # Define the number of initial points
    ni = 7

    # Define the start points
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Define the objective function
    fun = Analytical().fun_sphere

    # Initialize function control
    fun_control = fun_control_init(
        PREFIX = "test_spot_run",
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )

    # Initialize design control
    design_control = design_control_init(init_size=ni)

    # Create the Spot instance
    S = spot.Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
    )

    # Run the optimization
    S.run(X_start=X_start)

    # Check if the results are as expected
    assert S.X.shape[0] > 0, "The design matrix X should have more than 0 rows."
    assert S.y.shape[0] > 0, "The response vector y should have more than 0 elements."
    assert S.X.shape[0] == S.y.shape[0], "The design matrix X and response vector y should have the same number of rows."

    # Check if the minimum value in y is as expected
    assert np.min(np.abs(S.y)) == 0.0, "The minimum value in y should be 0.0."

    # Check if the corresponding x values are as expected
    min_index = np.argmin(S.y)
    assert np.allclose(S.X[min_index], [0.0, 0.0]), "The x values corresponding to the minimum y should be [0.0, 0.0]."

if __name__ == "__main__":
    pytest.main()