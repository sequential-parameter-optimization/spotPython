import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def test_set_var_name_from_fun_control():
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        var_name=["foo", "bar"]
    )
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # _set_var_name is called in __init__, but let's call it again for test
    spot._set_var_name()
    assert spot.var_name == ["foo", "bar"]

def test_set_var_name_default_names():
    fun_control = fun_control_init(
        lower=np.array([0, 0, 0]),
        upper=np.array([1, 1, 1]),
        var_name=None
    )
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._set_var_name()
    assert spot.var_name == ["x0", "x1", "x2"]

def test_set_var_name_empty_list():
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        var_name=[]
    )
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # If var_name is [], _set_var_name will not set default names (remains [])
    spot._set_var_name()
    assert spot.var_name == []

def test_set_var_name_after_changing_lower():
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        var_name=None
    )
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # Simulate changing lower after init
    spot.lower = np.array([0, 0, 0, 0])
    spot._set_var_name()
    assert spot.var_name == ["x0", "x1", "x2", "x3"]