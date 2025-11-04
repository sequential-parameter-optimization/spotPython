import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def dummy_fun(X, fun_control=None):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1)

def test_get_new_X0_returns_valid_shape():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    var_type = ['float', 'float']
    var_name = ['x1', 'x2']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name, n_points=2)
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    spot.X = np.array([[0.1, 0.2], [0.3, 0.4]])
    spot.y = dummy_fun(spot.X)
    spot.fit_surrogate()
    X0 = spot.get_new_X0()
    assert isinstance(X0, np.ndarray)
    assert X0.shape[1] == spot.k
    assert X0.shape[0] % spot.fun_repeats == 0
    assert np.all(X0 >= spot.lower)
    assert np.all(X0 <= spot.upper)

def test_get_new_X0_fallback(monkeypatch):
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    var_type = ['float', 'float']
    var_name = ['x1', 'x2']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name, n_points=2)
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    spot.X = np.array([[0.1, 0.2], [0.3, 0.4]])
    spot.y = dummy_fun(spot.X)
    spot.fit_surrogate()

    # Monkeypatch suggest_new_X to return empty array to trigger fallback
    def suggest_new_X_empty():
        return np.empty((0, spot.k))
    spot.suggest_new_X = suggest_new_X_empty

    X0 = spot.get_new_X0()
    assert isinstance(X0, np.ndarray)
    assert X0.shape[1] == spot.k
    assert np.all(X0 >= spot.lower)
    assert np.all(X0 <= spot.upper)