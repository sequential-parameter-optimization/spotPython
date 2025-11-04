import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def test_evaluate_initial_design_basic():
    # Simple quadratic function, single objective
    def fun(X, **kwargs):
        return np.sum(X**2, axis=1)
    fun_control = fun_control_init(lower=np.array([-1, -1]), upper=np.array([1, 1]), fun_evals=4)
    design_control = design_control_init(init_size=4)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    assert spot.X.shape[0] == spot.y.shape[0]
    assert spot.y.ndim == 1
    assert np.all(np.isfinite(spot.y))

def test_evaluate_initial_design_multi_objective_default():
    # Multi-objective: returns (n,2), default _mo2so should select first column
    def fun(X, **kwargs):
        return np.stack([np.sum(X, axis=1), np.prod(X, axis=1)], axis=1)
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]), fun_evals=3)
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    # y should be the first column of y_mo
    y_mo = fun(spot.X, fun_control=spot.fun_control)
    np.testing.assert_array_equal(spot.y, y_mo[:, 0])

def test_evaluate_initial_design_multi_objective_custom_fun():
    # Multi-objective with custom reduction function
    def fun(X, **kwargs):
        return np.stack([X[:, 0], X[:, 1]], axis=1)
    def mo2so(y_mo):
        return y_mo[:, 0] + 2 * y_mo[:, 1]
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]), fun_evals=2)
    fun_control["fun_mo2so"] = mo2so
    design_control = design_control_init(init_size=2)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    y_mo = fun(spot.X, fun_control=spot.fun_control)
    np.testing.assert_array_equal(spot.y, mo2so(y_mo))

def test_evaluate_initial_design_nan_removal():
    # Function returns NaN for some points, should be removed
    def fun(X, **kwargs):
        y = np.sum(X, axis=1)
        y[0] = np.nan
        return y
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]), fun_evals=3)
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    assert not np.isnan(spot.y).any()
    assert spot.X.shape[0] == spot.y.shape[0]
    # Should have removed at least one row
    assert spot.X.shape[0] == 2

def test_evaluate_initial_design_all_nan_raises():
    # All values are NaN, should raise ValueError
    def fun(X, **kwargs):
        return np.full(X.shape[0], np.nan)
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]), fun_evals=2)
    design_control = design_control_init(init_size=2)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    with pytest.raises(ValueError):
        spot.evaluate_initial_design()