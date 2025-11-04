import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def test_increment_counter_default():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    initial = spot._get_counter()
    spot._increment_counter()
    assert spot._get_counter() == initial + 1

def test_increment_counter_by_n():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._set_counter(5)
    spot._increment_counter(3)
    assert spot._get_counter() == 8

def test_increment_counter_initializes_if_missing():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    del spot.counter
    spot._increment_counter()
    assert spot._get_counter() == 1

def test_increment_counter_raises_on_non_int():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    with pytest.raises(ValueError):
        spot._increment_counter(1.5)
    with pytest.raises(ValueError):
        spot._increment_counter("abc")

def test_increment_counter_raises_on_negative():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    with pytest.raises(ValueError):
        spot._increment_counter(-2)
    with pytest.raises(ValueError):
        spot._increment_counter(0)