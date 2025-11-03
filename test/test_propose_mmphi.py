import numpy as np
import pytest
from spotpython.utils.sampling import propose_mmphi_intensive_minimizing_point, mmphi_intensive

def test_propose_mmphi_intensive_minimizing_point_basic():
    # 2D, 3 points, propose a 4th
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    new_point = propose_mmphi_intensive_minimizing_point(X, n_candidates=100, q=2, p=2, seed=42)
    assert new_point.shape == (1, 2)
    # The new point should not be a duplicate of existing points
    assert not any(np.allclose(new_point, x) for x in X)
    # Adding the new point should not make the criterion much worse (allow some tolerance)
    phi_before, _, _ = mmphi_intensive(X, q=2, p=2)
    phi_after, _, _ = mmphi_intensive(np.vstack([X, new_point]), q=2, p=2)
    assert phi_after < 2 * phi_before  # Allow some increase, but not arbitrarily large

def test_propose_mmphi_intensive_minimizing_point_bounds():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    lower = np.array([0.2, 0.2])
    upper = np.array([0.8, 0.8])
    new_point = propose_mmphi_intensive_minimizing_point(X, n_candidates=50, q=2, p=2, seed=1, lower=lower, upper=upper)
    assert np.all(new_point >= lower) and np.all(new_point <= upper)

def test_propose_mmphi_intensive_minimizing_point_highdim():
    X = np.random.rand(5, 10)
    new_point = propose_mmphi_intensive_minimizing_point(X, n_candidates=30, q=2, p=2, seed=123)
    assert new_point.shape == (1, 10)
    # Should not be a duplicate
    assert not any(np.allclose(new_point, x) for x in X)

def test_propose_mmphi_intensive_minimizing_point_single_point():
    X = np.array([[0.5, 0.5]])
    new_point = propose_mmphi_intensive_minimizing_point(X, n_candidates=10, q=2, p=2, seed=0)
    assert new_point.shape == (1, 2)
    # Should not be the same as the original point
    assert not np.allclose(new_point, X[0])