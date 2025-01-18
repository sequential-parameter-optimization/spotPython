import numpy as np
import pytest
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import spot
from spotpython.utils.init import fun_control_init, design_control_init


@pytest.fixture
def setup_spot():
    """
    PyTest Fixture for initializing Spot with given parameters.
    """
    ni = 7  # number of initial points
    ne = 20  # number of evaluations
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    fun = Analytical().fun_sphere
    fun_control = fun_control_init(
        PREFIX = "test_spot_run",
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        fun_evals=ne)
    design_control = design_control_init(init_size=ni)

    S = spot.Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
    )
    return S, X_start


def test_spot_run_shapes(setup_spot):
    """
    Test the shapes of S.X and S.y after running the Spot.run method.
    """
    S, X_start = setup_spot
    ne = S.fun_control["fun_evals"]
    exp_X_shape = (ne, 2)
    exp_y_shape = (ne,)

    S.run(X_start=X_start)

    # Assert shapes
    assert S.X.shape == exp_X_shape, f"Optimized S.X shape {S.X.shape} does not match expected shape {exp_X_shape}."
    assert S.y.shape == exp_y_shape, f"Optimized S.y shape {S.y.shape} does not match expected shape {exp_y_shape}."
