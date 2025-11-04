import pytest
import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

@pytest.fixture
def fun_control():
    return fun_control_init(
        lower=np.array([1, 2, 3]),
        upper=np.array([4, 5, 6]),
        seed=42
    )

def test_set_bounds_and_dim_sets_lower_upper_and_k(fun_control):
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    assert np.all(spot.lower == np.array([1, 2, 3]))
    assert np.all(spot.upper == np.array([4, 5, 6]))
    assert spot.k == 3

def test_set_bounds_and_dim_with_different_bounds():
    fc = fun_control_init(
        lower=np.array([-5, 0]),
        upper=np.array([5, 10]),
        seed=123
    )
    spot = Spot(fun=dummy_fun, fun_control=fc)
    assert np.all(spot.lower == np.array([-5, 0]))
    assert np.all(spot.upper == np.array([5, 10]))
    assert spot.k == 2

def test_set_bounds_and_dim_missing_lower_raises():
    fc = fun_control_init(
        upper=np.array([1, 2]),
        seed=1
    )
    # Remove lower key to simulate missing lower
    fc.pop("lower", None)
    with pytest.raises(AttributeError):
        Spot(fun=dummy_fun, fun_control=fc)