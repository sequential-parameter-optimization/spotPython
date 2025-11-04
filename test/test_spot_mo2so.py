import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def test_mo2so_default_behavior():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=lambda X, **kwargs: np.sum(X, axis=1), fun_control=fun_control)
    y_mo = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Should return first column by default
    y_so = spot._mo2so(y_mo)
    np.testing.assert_array_equal(y_so, np.array([1.0, 3.0]))

def test_mo2so_with_custom_fun():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    # Custom function: sum objectives
    fun_control["fun_mo2so"] = lambda y_mo: y_mo.sum(axis=1)
    spot = Spot(fun=lambda X, **kwargs: np.sum(X, axis=1), fun_control=fun_control)
    y_mo = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_so = spot._mo2so(y_mo)
    np.testing.assert_array_equal(y_so, np.array([3.0, 7.0]))

def test_mo2so_single_objective():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=lambda X, **kwargs: np.sum(X, axis=1), fun_control=fun_control)
    y_mo = np.array([5.0, 6.0])
    y_so = spot._mo2so(y_mo)
    np.testing.assert_array_equal(y_so, np.array([5.0, 6.0]))

def test_mo2so_empty_input():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=lambda X, **kwargs: np.sum(X, axis=1), fun_control=fun_control)
    y_mo = np.empty((0, 2))
    y_so = spot._mo2so(y_mo)
    assert y_so.size == 0