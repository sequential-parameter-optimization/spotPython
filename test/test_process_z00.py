import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

@pytest.fixture
def spot_instance():
    fun_control = fun_control_init(
        lower=np.array([0, 0, 0, 0]),
        upper=np.array([1, 1, 1, 1]),
        var_type=["float", "int", "int", "float"],
        PREFIX="pytest_process_z00"
    )
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.var_type = ["float", "int", "int", "float"]
    return spot

def test_process_z00_use_min(spot_instance):
    z00 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = spot_instance.process_z00(z00, use_min=True)
    # float: mean, int: min
    assert np.isclose(result[0], 3.0)
    assert result[1] == 2
    assert result[2] == 3
    assert np.isclose(result[3], 6.0)

def test_process_z00_use_max(spot_instance):
    z00 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = spot_instance.process_z00(z00, use_min=False)
    # float: mean, int: max
    assert np.isclose(result[0], 3.0)
    assert result[1] == 6
    assert result[2] == 7
    assert np.isclose(result[3], 6.0)

def test_process_z00_single_row(spot_instance):
    z00 = np.array([[10, 20, 30, 40]])
    result = spot_instance.process_z00(z00, use_min=True)
    assert np.isclose(result[0], 10.0)
    assert result[1] == 20
    assert result[2] == 30
    assert np.isclose(result[3], 40.0)

def test_process_z00_all_float(spot_instance):
    spot_instance.var_type = ["float", "float"]
    z00 = np.array([[1.5, 2.5], [3.5, 4.5]])
    result = spot_instance.process_z00(z00, use_min=True)
    assert np.allclose(result, [2.5, 3.5])

def test_process_z00_all_int(spot_instance):
    spot_instance.var_type = ["int", "int"]
    z00 = np.array([[1, 2], [3, 4]])
    result = spot_instance.process_z00(z00, use_min=True)
    assert result == [1, 2]
    result_max = spot_instance.process_z00(z00, use_min=False)
    assert result_max == [3, 4]