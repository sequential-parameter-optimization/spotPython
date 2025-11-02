import numpy as np
import pytest
from spotpython.fun.objectivefunctions import Analytical

def test_fun_michalewicz_global_minimum():
    fun = Analytical()
    # Known global minimum for d=2 is at approx [2.20, 1.57], value ≈ -1.8013
    X = np.array([[2.20, 1.57]])
    y = fun.fun_michalewicz(X)
    assert np.allclose(y, -1.8013, atol=1e-3)

def test_fun_michalewicz_shape_and_finiteness():
    fun = Analytical()
    X = np.array([[1.0, 2.0], [0.5, 1.0]])
    y = fun.fun_michalewicz(X)
    assert y.shape == (2,)
    assert np.all(np.isfinite(y))

def test_fun_michalewicz_noise():
    fun = Analytical(sigma=0.5, seed=42)
    X = np.array([[2.20, 1.57], [1.0, 2.0]])
    y = fun.fun_michalewicz(X)
    # Should not be exactly the noiseless value
    assert not np.allclose(y, fun.fun_michalewicz(X, fun_control={"sigma": 0.0, "seed": 42}))
    assert y.shape == (2,)