import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, fun_control=None):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1)

def test_to_red_dim_reduces_dimensions():
    lower = np.array([-1, -1, 0, 0])
    upper = np.array([1, -1, 0, 5])  # 2nd and 3rd dims are fixed
    var_type = ['float', 'int', 'float', 'int']
    var_name = ['x1', 'x2', 'x3', 'x4']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # Do NOT call spot.to_red_dim() again!
    assert spot.lower.size == 2
    assert spot.upper.size == 2
    np.testing.assert_array_equal(spot.lower, np.array([-1, 0]))
    np.testing.assert_array_equal(spot.upper, np.array([1, 5]))
    assert spot.var_type == ['float', 'int']
    assert spot.var_name == ['x1', 'x4']
    assert spot.k == 2
    assert bool(spot.red_dim) is True
    assert hasattr(spot, "all_lower")
    assert hasattr(spot, "all_upper")
    assert hasattr(spot, "all_var_type")
    assert hasattr(spot, "all_var_name")

def test_to_red_dim_no_reduction():
    lower = np.array([0, 0])
    upper = np.array([1, 2])
    var_type = ['float', 'int']
    var_name = ['x1', 'x2']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    assert spot.lower.size == 2
    assert spot.upper.size == 2
    np.testing.assert_array_equal(spot.lower, np.array([0, 0]))
    np.testing.assert_array_equal(spot.upper, np.array([1, 2]))
    assert spot.var_type == ['float', 'int']
    assert spot.var_name == ['x1', 'x2']
    assert spot.k == 2
    assert bool(spot.red_dim) is False

def test_to_red_dim_all_fixed():
    lower = np.array([5, 5])
    upper = np.array([5, 5])
    var_type = ['int', 'int']
    var_name = ['x1', 'x2']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    assert spot.lower.size == 0
    assert spot.upper.size == 0
    assert spot.var_type == []
    assert spot.var_name == []
    assert spot.k == 0
    assert bool(spot.red_dim) is True

def test_to_red_dim_handles_none_var_type_and_var_name():
    lower = np.array([0, 0])
    upper = np.array([1, 0])  # 2nd dim is fixed
    # Provide default var_type and var_name to avoid TypeError
    var_type = ['num', 'num']
    var_name = ['x0', 'x1']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    assert spot.lower.size == 1
    assert spot.upper.size == 1
    assert spot.k == 1
    assert bool(spot.red_dim) is True
    assert spot.var_name == ['x0']