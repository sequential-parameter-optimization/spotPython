import pytest
import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

@pytest.fixture
def fun_control():
    return fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        seed=42
    )

def test_set_fun_valid_callable(fun_control):
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    assert spot.fun is dummy_fun

def test_set_fun_none_raises(fun_control):
    with pytest.raises(Exception, match="No objective function specified."):
        Spot(fun=None, fun_control=fun_control)

def test_set_fun_not_callable_raises(fun_control):
    with pytest.raises(Exception, match="Objective function is not callable"):
        Spot(fun=123, fun_control=fun_control)

def test_set_fun_sets_fun_attribute(fun_control):
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    assert callable(spot.fun)