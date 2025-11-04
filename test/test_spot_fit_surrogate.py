import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init, surrogate_control_init

def test_fit_surrogate_basic():
    def fun(X, **kwargs):
        return np.sum(X**2, axis=1)
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        fun_evals=5
    )
    design_control = design_control_init(init_size=5)
    surrogate_control = surrogate_control_init()
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    spot.fit_surrogate()
    X_test = np.array([[0.0, 0.0]])
    y_pred = spot.surrogate.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape[0] == 1

def test_fit_surrogate_warns_on_shape_mismatch(caplog):
    def fun(X, **kwargs):
        return np.sum(X, axis=1)
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        fun_evals=3
    )
    design_control = design_control_init(init_size=3)
    surrogate_control = surrogate_control_init()
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    spot.y = spot.y[:-1]
    # Specify the logger name used in spot.py
    with caplog.at_level("WARNING", logger="spotpython.spot.spot"):
        spot.fit_surrogate()
        assert "X and y have different sizes" in caplog.text

def test_fit_surrogate_with_many_points_triggers_selection(monkeypatch):
    def fun(X, **kwargs):
        return np.sum(X, axis=1)
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        fun_evals=20
    )
    design_control = design_control_init(init_size=20)
    surrogate_control = surrogate_control_init(max_surrogate_points=5, use_nystrom=False)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    called = {}
    def fake_select_distant_points(X, y, k):
        called['called'] = True
        return X[:k], y[:k]
    monkeypatch.setattr("spotpython.spot.spot.select_distant_points", fake_select_distant_points)
    spot.fit_surrogate()
    assert called.get('called', False)