import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def test_get_counter_initial():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    assert spot._get_counter() == 0

def test_get_counter_after_set():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._set_counter(5)
    assert spot._get_counter() == 5

def test_get_counter_after_increment():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._increment_counter(3)
    assert spot._get_counter() == 3

def test_get_counter_none_value():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.counter = None
    assert spot._get_counter() == 0

def test_get_counter_negative_value():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.counter = -7
    # _get_counter should just return the value, even if negative (no check in _get_counter)
    assert spot._get_counter() == -7