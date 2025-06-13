import numpy as np
import pytest
from spotpython.utils.sampling import mmphi_intensive
from spotpython.utils import sampling
from scipy.spatial.distance import pdist
from spotpython.utils import sampling

class DummyJD:
    """Helper to monkeypatch jd for controlled testing."""
    def __init__(self, J, d):
        self.J = J
        self.d = d
    def __call__(self, X, p=2.0):
        return self.J, self.d

def test_mmphi_intensive_basic(monkeypatch):
    # Use a simple 2D square, distances are all 1 or sqrt(2)
    X = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    # Patch jd to use real distances for this test
    orig_jd = getattr(sampling, "jd", None)
    def real_jd(X, p=2.0):
        dists = pdist(X, metric="minkowski", p=p)
        # Count unique distances and their multiplicities
        vals, counts = np.unique(np.round(dists, 8), return_counts=True)
        return counts, vals
    monkeypatch.setattr(sampling, "jd", real_jd)
    val, J, d = mmphi_intensive(X, q=2.0, p=2.0)
    assert np.isscalar(val)
    assert val > 0
    if orig_jd:
        monkeypatch.setattr(sampling, "jd", orig_jd)

def test_mmphi_intensive_duplicates(monkeypatch):
    # All points identical: should return np.inf
    X = np.ones((4, 2))
    monkeypatch.setattr(sampling, "jd", lambda X, p=2.0: (np.array([]), np.array([])))
    val, J, d = mmphi_intensive(X)
    assert val == np.inf

def test_mmphi_intensive_too_few_points():
    # Only one point: should return np.inf
    X = np.array([[0.5, 0.5]])
    val, J, d = mmphi_intensive(X)
    assert val == np.inf
