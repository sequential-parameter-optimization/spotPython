import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def make_spot():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # Simulate initial design and evaluation
    spot.X = np.array([[0, 0], [1, 1], [0.5, 0.5]])
    spot.y = np.array([2.0, 1.0, 0.5])
    return spot

def test_update_success_rate_improvement():
    spot = make_spot()
    # New y value is better than all previous (should be counted as success)
    spot._update_success_rate(np.array([0.1]))
    assert spot.success_rate == 1.0

def test_update_success_rate_no_improvement():
    spot = make_spot()
    # New y value is worse than best so far (should be counted as failure)
    spot._update_success_rate(np.array([2.5]))
    assert spot.success_rate == 0.0

def test_update_success_rate_mixed():
    spot = make_spot()
    # Add a sequence: first is improvement, second is not
    spot._update_success_rate(np.array([0.4, 0.6]))
    # First is success, second is not (success history: [1, 0])
    assert spot.success_rate == 0.5

def test_update_success_rate_window_size():
    spot = make_spot()
    spot.window_size = 3
    # Add 4 values, only last 3 should be considered
    spot._update_success_rate(np.array([0.4, 0.3, 0.2, 0.1]))
    # All are improvements, so last 3 are successes
    assert spot.success_rate == 1.0
    # Add a non-improvement, now last 3: [1, 1, 0]
    spot._update_success_rate(np.array([0.5]))
    assert spot.success_rate == 2/3

def test_update_success_rate_empty_y():
    spot = make_spot()
    spot.y = np.array([])
    # Should treat best_y as inf, so any value is an improvement
    spot._update_success_rate(np.array([5.0]))
    assert spot.success_rate == 1.0