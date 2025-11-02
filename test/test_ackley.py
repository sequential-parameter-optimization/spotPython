import numpy as np
import pytest
from spotpython.fun.objectivefunctions import Analytical

def test_fun_ackley_basic():
    fun = Analytical()
    # Test at the global minimum (should be close to 0)
    X = np.zeros((3, 5))
    y = fun.fun_ackley(X)
    assert np.allclose(y, 0, atol=1e-7)

def test_fun_ackley_typical_domain():
    fun = Analytical()
    # Test at a random point in the typical domain
    X = np.array([[1.0, 2.0, 3.0], [-10.0, 0.0, 10.0]])
    y = fun.fun_ackley(X)
    assert y.shape == (2,)
    assert np.all(np.isfinite(y))

def test_fun_ackley_noise():
    fun = Analytical(sigma=0.5, seed=42)
    X = np.zeros((5, 2))
    y = fun.fun_ackley(X)
    # Should not be exactly zero due to noise
    assert not np.allclose(y, 0)
    assert y.shape == (5,)