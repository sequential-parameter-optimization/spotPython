# test/test_spot_update_design.py

import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def make_spot_instance(noise=False, ocba_delta=0):
    fun = lambda X, **kwargs: np.sum(X**2, axis=1)
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        fun_evals=10,
        noise=noise,
        ocba_delta=ocba_delta,
    )
    design_control = design_control_init(init_size=4)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design()
    spot.update_stats()
    spot.fit_surrogate()
    return spot

def test_update_design_adds_points():
    spot = make_spot_instance()
    X_shape_before = spot.X.shape
    y_size_before = spot.y.size
    spot.update_design()
    # At least one new point should be added
    assert spot.X.shape[0] > X_shape_before[0]
    assert spot.y.size > y_size_before
    # Shape should match
    assert spot.X.shape[0] == spot.y.size

def test_update_design_multiple_calls_grow_design():
    spot = make_spot_instance()
    initial_n = spot.X.shape[0]
    for _ in range(3):
        spot.update_design()
    assert spot.X.shape[0] > initial_n

def test_update_design_with_noise_and_ocba():
    spot = make_spot_instance(noise=True, ocba_delta=1)
    X_shape_before = spot.X.shape
    y_size_before = spot.y.size
    spot.update_stats()
    spot.fit_surrogate()
    spot.update_design()
    # OCBA should add at least ocba_delta points
    assert spot.X.shape[0] >= X_shape_before[0] + spot.ocba_delta
    assert spot.y.size >= y_size_before + spot.ocba_delta

def test_update_design_preserves_feature_count():
    spot = make_spot_instance()
    spot.update_design()
    assert spot.X.shape[1] == 2

def test_update_design_handles_nan_gracefully():
    # Use a function that returns NaN for some points
    def fun_nan(X, **kwargs):
        y = np.sum(X**2, axis=1)
        y[0] = np.nan
        return y
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        fun_evals=10,
    )
    design_control = design_control_init(init_size=4)
    spot = Spot(fun=fun_nan, fun_control=fun_control, design_control=design_control)
    spot.initialize_design()
    spot.update_stats()
    spot.fit_surrogate()
    spot.update_design()
    # No NaN in y after update
    assert not np.isnan(spot.y).any()