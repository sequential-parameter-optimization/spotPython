import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def make_spot_with_success_rate(success_rate):
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.success_rate = success_rate
    return spot

def test_get_success_rate_default():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    assert spot._get_success_rate() == 0.0

def test_get_success_rate_set_value():
    spot = make_spot_with_success_rate(0.75)
    assert spot._get_success_rate() == 0.75

def test_get_success_rate_none():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.success_rate = None
    assert spot._get_success_rate() == 0.0

def test_get_success_rate_after_update():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.y = np.array([2.0, 1.0, 0.5])
    spot._update_success_rate(np.array([0.4, 0.6]))
    # First is success (0.4 < 0.5), second is not (0.6 > 0.4)
    assert spot._get_success_rate() == 0.5