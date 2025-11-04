import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def quadratic_fun(X, **kwargs):
    return np.sum(X**2, axis=1)

@pytest.fixture
def spot_instance():
    fun_control = fun_control_init(
        lower=np.array([-1, -1, -1]),
        upper=np.array([1, 1, 1]),
        PREFIX="pytest_chg",
        save_result=False,
        save_experiment=False,
    )
    spot = Spot(fun=quadratic_fun, fun_control=fun_control)
    return spot

def test_chg_list(spot_instance):
    z0 = [1, 2, 3]
    result = spot_instance.chg(x=10, y=20, z0=z0.copy(), i=0, j=2)
    assert result == [10, 2, 20]

def test_chg_numpy_array(spot_instance):
    z0 = np.array([1, 2, 3])
    result = spot_instance.chg(x=5, y=7, z0=z0.copy(), i=1, j=0)
    assert np.array_equal(result, np.array([7, 5, 3]))

def test_chg_same_index(spot_instance):
    z0 = [1, 2, 3]
    result = spot_instance.chg(x=9, y=8, z0=z0.copy(), i=1, j=1)
    assert result == [1, 8, 3]  # last assignment wins

def test_chg_does_not_modify_original(spot_instance):
    z0 = [1, 2, 3]
    z0_copy = z0.copy()
    _ = spot_instance.chg(x=4, y=5, z0=z0_copy, i=0, j=2)
    assert z0 == [1, 2, 3]