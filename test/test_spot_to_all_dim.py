import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, fun_control=None):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1)

def test_to_all_dim_reconstructs_full_dim():
    # 2nd and 3rd dims are fixed
    lower = np.array([-1, -1, 0, 0])
    upper = np.array([1, -1, 0, 5])
    var_type = ['float', 'int', 'float', 'int']
    var_name = ['x1', 'x2', 'x3', 'x4']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # After reduction, only dims 0 and 3 remain
    X_reduced = np.array([[0.5, 2.0], [1.0, 3.0]])
    X_full = spot.to_all_dim(X_reduced)
    # Should reconstruct full dimension: [-1, -1, 0, 0] + [1, -1, 0, 5]
    expected = np.array([
        [0.5, -1, 0, 2.0],
        [1.0, -1, 0, 3.0]
    ])
    np.testing.assert_array_equal(X_full, expected)

def test_to_all_dim_no_fixed_dimensions():
    lower = np.array([0, 0])
    upper = np.array([1, 2])
    var_type = ['float', 'int']
    var_name = ['x1', 'x2']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # No reduction, so to_all_dim should return input unchanged
    X = np.array([[0.1, 0.2], [0.3, 1.5]])
    X_full = spot.to_all_dim(X)
    np.testing.assert_array_equal(X_full, X)

def test_to_all_dim_all_fixed_dimensions():
    lower = np.array([5, 5])
    upper = np.array([5, 5])
    var_type = ['int', 'int']
    var_name = ['x1', 'x2']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # All dims fixed, so reduced X is shape (n,0), to_all_dim should fill with fixed values
    X_reduced = np.empty((3, 0))
    X_full = spot.to_all_dim(X_reduced)
    expected = np.full((3, 2), 5)
    np.testing.assert_array_equal(X_full, expected)

def test_to_all_dim_handles_single_sample():
    lower = np.array([-1, 0, 0])
    upper = np.array([1, 0, 5])
    var_type = ['float', 'int', 'int']
    var_name = ['x1', 'x2', 'x3']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # Only dims 0 and 2 remain
    X_reduced = np.array([[0.7, 4.2]])
    X_full = spot.to_all_dim(X_reduced)
    expected = np.array([[0.7, 0, 4.2]])
    np.testing.assert_array_equal(X_full, expected)