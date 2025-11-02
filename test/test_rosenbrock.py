import numpy as np
import pytest
from spotpython.fun.objectivefunctions import Analytical

def test_fun_rosenbrock_global_minimum():
    fun = Analytical()
    # Global minimum at x_i = 1 for all i, f(x*) = 0
    X = np.ones((4, 3))
    y = fun.fun_rosenbrock(X)
    assert np.allclose(y, 0, atol=1e-8)

def test_fun_rosenbrock_typical_values():
    fun = Analytical()
    X = np.array([[0, 0, 0], [1, 2, 3], [-1, -1, -1]])
    y = fun.fun_rosenbrock(X)
    assert y.shape == (3,)
    assert np.all(np.isfinite(y))

def test_fun_rosenbrock_noise():
    fun = Analytical(sigma=0.5, seed=123)
    X = np.ones((5, 2))
    y = fun.fun_rosenbrock(X)
    # Should not be exactly zero due to noise
    assert not np.allclose(y, 0)
    assert y.shape == (5,)