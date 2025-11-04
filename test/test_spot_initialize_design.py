import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def dummy_fun(X, fun_control=None):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1)

def test_initialize_design_basic():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper, fun_evals=5)
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design()
    # X and y should be initialized and have correct shapes
    assert spot.X is not None
    assert spot.y is not None
    assert spot.X.shape[1] == 2
    assert spot.X.shape[0] == design_control["init_size"]
    assert spot.y.shape[0] == design_control["init_size"]

def test_initialize_design_with_X_start():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper, fun_evals=5)
    design_control = design_control_init(init_size=2)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    X_start = np.array([[0.5, 0.5], [0.1, 0.9]])
    spot.initialize_design(X_start=X_start)
    # X should contain both X_start and generated points
    assert spot.X.shape[1] == 2
    assert spot.X.shape[0] == design_control["init_size"] + X_start.shape[0]
    # y should match X in number of rows
    assert spot.y.shape[0] == spot.X.shape[0]
    # X_start rows should be present in X
    assert all(any(np.allclose(x, row, atol=1e-8) for x in spot.X) for row in X_start)

def test_initialize_design_wrong_shape_X_start():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper, fun_evals=5)
    design_control = design_control_init(init_size=2)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    # X_start with wrong shape (should be ignored)
    X_start = np.array([0.5, 0.5, 0.5])
    spot.initialize_design(X_start=X_start)
    # Should still have correct number of rows
    assert spot.X.shape[1] == 2
    assert spot.X.shape[0] == design_control["init_size"]
    assert spot.y.shape[0] == design_control["init_size"]

def test_initialize_design_zero_init_size():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper, fun_evals=2)
    design_control = design_control_init(init_size=0)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    X_start = np.array([[0.2, 0.8]])
    spot.initialize_design(X_start=X_start)
    assert spot.X.shape == (1, 2)
    assert spot.y.shape == (1,)
    np.testing.assert_array_equal(spot.X, X_start)