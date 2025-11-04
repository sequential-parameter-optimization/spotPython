import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def test_set_de_bounds_basic():
    fun_control = fun_control_init(lower=np.array([-1, -2]), upper=np.array([1, 2]))
    S = Spot(fun=lambda X, fun_control=None: np.sum(X, axis=1), fun_control=fun_control)
    S.set_de_bounds()
    assert hasattr(S, "de_bounds")
    assert S.de_bounds == [[-1, 1], [-2, 2]]

def test_set_de_bounds_single_dim():
    fun_control = fun_control_init(lower=np.array([0]), upper=np.array([10]))
    S = Spot(fun=lambda X, fun_control=None: np.sum(X, axis=1), fun_control=fun_control)
    S.set_de_bounds()
    assert S.de_bounds == [[0, 10]]

def test_set_de_bounds_after_dim_reduction():
    fun_control = fun_control_init(lower=np.array([-1, 0, 5]), upper=np.array([1, 0, 10]))
    S = Spot(fun=lambda X, fun_control=None: np.sum(X, axis=1), fun_control=fun_control)
    # Simulate dimension reduction (second dim fixed)
    S.to_red_dim()
    S.set_de_bounds()
    # Only variable dims remain: [-1, 5] to [1, 10]
    assert S.de_bounds == [[-1, 1], [5, 10]]

def test_set_de_bounds_types():
    fun_control = fun_control_init(lower=np.array([-1.5, 2.5]), upper=np.array([1.5, 3.5]))
    S = Spot(fun=lambda X, fun_control=None: np.sum(X, axis=1), fun_control=fun_control)
    S.set_de_bounds()
    assert all(isinstance(b, list) and len(b) == 2 for b in S.de_bounds)
    assert np.allclose(S.de_bounds[0], [-1.5, 1.5])
    assert np.allclose(S.de_bounds[1], [2.5, 3.5])