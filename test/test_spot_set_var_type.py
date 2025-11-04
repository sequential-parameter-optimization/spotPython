import pytest
import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, fun_control=None):
    return np.sum(X, axis=1)

def make_spot_with_var_type(var_type, k=2):
    fun_control = fun_control_init(
        lower=np.zeros(k),
        upper=np.ones(k),
        var_type=var_type,
    )
    return Spot(fun=dummy_fun, fun_control=fun_control)

def test_var_type_all_num():
    spot = make_spot_with_var_type(["num", "num"])
    assert spot.var_type == ["num", "num"]

def test_var_type_int_and_float():
    spot = make_spot_with_var_type(["int", "float"])
    assert spot.var_type == ["int", "float"]

def test_var_type_factor():
    spot = make_spot_with_var_type(["factor", "factor"])
    assert spot.var_type == ["factor", "factor"]

def test_var_type_shorter_than_k():
    spot = make_spot_with_var_type(["num"], k=3)
    assert spot.var_type == ["num", "num", "num"]

def test_var_type_invalid_raises():
    with pytest.raises(ValueError):
        make_spot_with_var_type(["num", "invalid"])

def test_var_type_superset_num():
    # "num" is allowed and is a superset of "int" and "float"
    spot = make_spot_with_var_type(["num", "int", "float"], k=3)
    assert spot.var_type == ["num", "int", "float"]