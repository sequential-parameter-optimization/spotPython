import numpy as np
import pytest
from numpy.testing import assert_allclose

from spotpython.surrogate.kriging import Kriging


def _make_test_data(k=2, n=15, seed=0):
    """Helper to create synthetic test data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, k))
    # Handle 1D case without indexing second dimension
    y = np.sin(X[:, 0]) + (0.1 * X[:, 1] if k > 1 else 0.0)
    return X, np.asarray(y, dtype=float)


def _setup_model(X, y, method="regression", lambda_lin=1e-2, m=None):
    """Helper to setup a minimal working model."""
    model = Kriging(method=method, use_nystrom=True, nystrom_m=m)
    model.X_ = np.asarray(X)
    model.y_ = np.asarray(y).ravel()
    model.n, model.k = model.X_.shape
    model._set_variable_types()
    model.n_theta = model.k
    model.theta = np.full(model.k, -1.0)  # stable default
    model.lambda_lin_ = float(lambda_lin)
    model.method = method  # Set method explicitly
    return model


def test_likelihood_nystrom_regression_basic():
    """Test basic likelihood computation for regression case."""
    X, y = _make_test_data(k=2, n=10, seed=1)
    model = _setup_model(X, y, method="regression", m=4)
    
    # x = [logtheta..., log10(lambda)]
    x = np.array([-1.0, -1.0, -2.0])  # 2 thetas + 1 lambda
    negll, P, U = model._likelihood_nystrom(x)
    
    assert np.isfinite(negll)
    assert P is None and U is None
    assert model.M_chol_ is not None
    assert model.lambda_lin_ == 1e-2  # 10**(-2)


def test_likelihood_nystrom_interpolation():
    """Test interpolation case (no lambda parameter)."""
    X, y = _make_test_data(k=2, n=12, seed=2)
    model = _setup_model(X, y, method="interpolation", m=5)
    
    x = np.array([-1.0, -1.0])  # only thetas
    negll, _, _ = model._likelihood_nystrom(x)
    
    assert np.isfinite(negll)
    assert model.lambda_lin_ == model.eps


def test_likelihood_nystrom_with_optim_p():
    """Test likelihood computation with p-parameter optimization."""
    X, y = _make_test_data(k=2, n=10, seed=3)
    model = Kriging(method="regression", use_nystrom=True, nystrom_m=4, optim_p=True)
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    model.n_theta = model.k
    
    # x = [logtheta..., log10(lambda), p...]
    x = np.array([-1.0, -1.0, -2.0, 1.5, 1.5])
    negll, _, _ = model._likelihood_nystrom(x)
    
    assert np.isfinite(negll)
    assert np.allclose(model.p_val, np.array([1.5, 1.5]))


def test_likelihood_nystrom_bad_sigma_returns_penalty():
    """Test that negative/invalid sigma returns penalty value."""
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0])  # Degenerate case -> zero variance
    model = _setup_model(X, y, method="regression", m=2)
    
    x = np.array([-1.0, -1.0, -2.0])
    negll, P, U = model._likelihood_nystrom(x)
    
    assert negll == model.penalty
    assert P is None and U is None



def test_likelihood_nystrom_determinant_terms():
    """Test the determinant computation via matrix lemma."""
    X, y = _make_test_data(k=1, n=8, seed=4)  # 1D for simplicity
    model = _setup_model(X, y, method="regression", m=3)
    
    x = np.array([-1.0, -2.0])  # 1 theta + 1 lambda
    model._nystrom_setup()  # Ensure Nyström structures exist
    negll, _, _ = model._likelihood_nystrom(x)
    
    # Check components are available
    assert model.W_chol_ is not None
    assert model.M_chol_ is not None
    assert model.C_ is not None
    
    # Matrix shapes
    assert model.W_chol_.shape == (3, 3)  # m x m
    assert model.C_.shape == (8, 3)  # n x m


def test_likelihood_nystrom_numerical_stability():
    """Test numerical stability with different lambda values."""
    X, y = _make_test_data(k=2, n=15, seed=5)
    model = _setup_model(X, y, method="regression", m=6)
    
    # Test range of lambda values
    for log_lambda in [-6, -3, 0, 3]:
        x = np.array([-1.0, -1.0, float(log_lambda)])
        negll, _, _ = model._likelihood_nystrom(x)
        assert np.isfinite(negll)



def test_likelihood_nystrom_method_validation():
    """Test method validation in likelihood computation."""
    X, y = _make_test_data(k=2, n=10, seed=6)
    
    # Directly test valid methods
    for method in ["interpolation", "regression", "reinterpolation"]:
        model = _setup_model(X, y, method=method, m=4)
        x = np.array([-1.0, -1.0, -2.0])
        negll, _, _ = model._likelihood_nystrom(x)
        assert np.isfinite(negll)

def test_likelihood_nystrom_dimension_checks():
    """Test input dimension validation."""
    X, y = _make_test_data(k=3, n=12, seed=7)
    model = _setup_model(X, y, method="regression", m=5)
    model._nystrom_setup()
    
    # Correct number of parameters (3 thetas + 1 lambda)
    x_correct = np.array([-1.0, -1.0, -1.0, -2.0])
    negll, _, _ = model._likelihood_nystrom(x_correct)
    assert np.isfinite(negll)
    
    # Test dimension validation by checking returned penalty
    cases = [
        np.array([-1.0, -1.0]),  # Too few parameters
        np.array([-1.0, -1.0, -2.0]),  # Missing one theta
        np.array([-1.0, -1.0, -1.0, -2.0, -1.0])  # Too many parameters
    ]
    
    for x_wrong in cases:
        negll, _, _ = model._likelihood_nystrom(x_wrong)
        assert negll == model.penalty, f"Expected penalty value for wrong input dimension {x_wrong.size}"