import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def test_set_counter_valid():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._set_counter(7)
    assert spot.counter == 7
    assert spot._get_counter() == 7

def test_set_counter_zero():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._set_counter(0)
    assert spot.counter == 0
    assert spot._get_counter() == 0

def test_set_counter_negative_raises():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    with pytest.raises(ValueError):
        spot._set_counter(-1)

def test_set_counter_non_integer_raises():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    with pytest.raises(ValueError):
        spot._set_counter(3.5)
    with pytest.raises(ValueError):
        spot._set_counter("abc")