import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, fun_control=None):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1)

def test_store_mo_initial_assignment():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.y_mo = None
    y_mo = np.array([[1.0, 2.0], [3.0, 4.0]])
    spot._store_mo(y_mo)
    assert spot.y_mo is not None
    np.testing.assert_array_equal(spot.y_mo, y_mo)

def test_store_mo_appends_rows():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.y_mo = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_mo_new = np.array([[5.0, 6.0], [7.0, 8.0]])
    spot._store_mo(y_mo_new)
    expected = np.vstack((np.array([[1.0, 2.0], [3.0, 4.0]]), y_mo_new))
    np.testing.assert_array_equal(spot.y_mo, expected)

def test_store_mo_shape_mismatch_raises():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.y_mo = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_mo_bad = np.array([[5.0, 6.0, 7.0]])
    try:
        spot._store_mo(y_mo_bad)
    except ValueError as e:
        assert "does not match the number of columns" in str(e)
    else:
        assert False, "Expected ValueError due to shape mismatch"