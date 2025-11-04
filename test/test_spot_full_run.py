import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def dummy_fun(X, fun_control=None):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)

def test_run_basic_execution():
    lower = np.array([-1, -1])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper, fun_evals=5)
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    result = spot.run()
    assert isinstance(result, Spot)
    # After run, X and y should be set and have at least fun_evals rows
    assert spot.X is not None
    assert spot.y is not None
    assert spot.X.shape[0] >= fun_control["fun_evals"]
    assert spot.y.shape[0] >= fun_control["fun_evals"]
    # min_y and min_X should be set
    assert hasattr(spot, "min_y")
    assert hasattr(spot, "min_X")

def test_run_with_initial_design():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper, fun_evals=4)
    design_control = design_control_init(init_size=2)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    X_start = np.array([[0, 0], [1, 1]])
    result = spot.run(X_start=X_start)
    assert isinstance(result, Spot)
    # X should contain the initial design
    assert np.any(np.all(spot.X == [0, 0], axis=1))
    assert np.any(np.all(spot.X == [1, 1], axis=1))
    assert spot.y.shape[0] >= fun_control["fun_evals"]

def test_run_sets_min_y_and_min_X():
    lower = np.array([-2, -2])
    upper = np.array([2, 2])
    fun_control = fun_control_init(lower=lower, upper=upper, fun_evals=6)
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    spot.run()
    # min_y should be the minimum of y
    assert np.isclose(spot.min_y, np.min(spot.y))
    # min_X should be the row in X corresponding to min_y
    idx = np.argmin(spot.y)
    np.testing.assert_array_equal(spot.min_X, spot.X[idx])