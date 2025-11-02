import numpy as np
import pytest
from spotpython.surrogate.kriging import Kriging

def _make_test_data(k=2, n=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, k))
    y = np.sin(X[:, 0]) + (0.1 * X[:, 1] if k > 1 else 0.0)
    return X, y

def test_max_likelihood_exact_regression():
    """Test max_likelihood() finds reasonable parameters (exact, regression)."""
    X, y = _make_test_data(k=2, n=10, seed=1)
    model = Kriging(method="regression")
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    model.n_theta = model.k

    bounds = [(-3, 2), (-3, 2), (-6, 0)]  # theta1, theta2, log10(lambda)
    params, negll = model.max_likelihood(bounds)
    assert params.shape == (3,)
    assert np.isfinite(negll)
    # Check that likelihood at optimum is not the penalty
    assert negll < model.penalty

def test_max_likelihood_nystrom_regression():
    """Test max_likelihood() works with Nyström approximation."""
    X, y = _make_test_data(k=2, n=12, seed=2)
    model = Kriging(method="regression", use_nystrom=True, nystrom_m=5)
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    model.n_theta = model.k

    bounds = [(-3, 2), (-3, 2), (-6, 0)]
    params, negll = model.max_likelihood(bounds)
    assert params.shape == (3,)
    assert np.isfinite(negll)
    assert negll < model.penalty

def test_max_likelihood_interpolation():
    """Test max_likelihood() for interpolation (no lambda)."""
    X, y = _make_test_data(k=1, n=8, seed=3)
    model = Kriging(method="interpolation")
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    model.n_theta = model.k

    bounds = [(-3, 2)]
    params, negll = model.max_likelihood(bounds)
    assert params.shape == (1,)
    assert np.isfinite(negll)
    assert negll < model.penalty

def test_max_likelihood_with_optim_p():
    """Test max_likelihood() with p-parameter optimization."""
    X, y = _make_test_data(k=2, n=10, seed=4)
    model = Kriging(method="regression", optim_p=True, n_p=2)
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    model.n_theta = model.k

    bounds = [(-3, 2), (-3, 2), (-6, 0), (1, 2), (1, 2)]  # thetas, lambda, p1, p2
    params, negll = model.max_likelihood(bounds)
    assert params.shape == (5,)
    assert np.isfinite(negll)
    assert negll < model.penalty

def test_max_likelihood_invalid_bounds():
    """Test max_likelihood() raises or returns penalty for invalid bounds."""
    X, y = _make_test_data(k=2, n=8, seed=5)
    model = Kriging(method="regression")
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    model.n_theta = model.k

    # Bounds that are too short
    bounds = [(-3, 2)]
    with pytest.raises(Exception):
        model.max_likelihood(bounds)