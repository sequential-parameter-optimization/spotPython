import numpy as np
import pytest
from numpy.testing import assert_allclose

from spotpython.surrogate.kriging import Kriging


def setup_minimal_model(X, y, *, method="regression", lambda_lin=1e-2, m=None, seed=123):
    """Helper to set up a minimal working model for testing woodbury solve."""
    model = Kriging(method=method, use_nystrom=True, nystrom_m=m, nystrom_seed=seed)
    model.X_ = np.asarray(X)
    model.y_ = np.asarray(y).ravel()
    model.n, model.k = model.X_.shape
    model._set_variable_types()
    model.n_theta = model.k
    model.theta = np.full(model.k, -1.0)  # stable default theta
    model.lambda_lin_ = float(lambda_lin)
    model._nystrom_setup()
    
    # Compute M = W + (1/λ)C^TC and its Cholesky
    W = model.W_chol_ @ model.W_chol_.T
    CtC = model.C_.T @ model.C_
    M = W + CtC / model.lambda_lin_
    try:
        model.M_chol_ = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        model.M_chol_ = np.linalg.cholesky(M + 1e-10 * np.eye(M.shape[0]))
    return model


def test_woodbury_solve_input_validation():
    """Test input validation and error handling."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    y = np.array([0.0, 1.0, 0.5])
    model = setup_minimal_model(X, y, m=2)
    
    # Wrong size vector
    with pytest.raises(ValueError):
        model._woodbury_solve(np.ones(model.n + 1))
    
    # Missing required state
    model_bad = setup_minimal_model(X, y, m=2)
    model_bad.M_chol_ = None
    with pytest.raises(RuntimeError):
        model_bad._woodbury_solve(np.ones(model.n))


def test_woodbury_solve_linearity():
    """Test that woodbury solve respects linear combinations."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 2.0])
    model = setup_minimal_model(X, y, lambda_lin=0.1, m=2)
    
    rng = np.random.default_rng(42)
    v1 = rng.normal(size=model.n)
    v2 = rng.normal(size=model.n)
    a, b = 2.5, -1.2
    
    # Test a*R^{-1}v1 + b*R^{-1}v2 = R^{-1}(a*v1 + b*v2)
    left = a * model._woodbury_solve(v1) + b * model._woodbury_solve(v2)
    right = model._woodbury_solve(a * v1 + b * v2)
    assert_allclose(left, right, rtol=1e-10, atol=1e-12)


def test_woodbury_solve_small_lambda_limit():
    """Test behavior as λ approaches zero (interpolation regime)."""
    X = np.array([[0.0], [0.5], [1.0]])
    y = np.array([0.0, 0.5, 1.0])
    
    # Very small lambda
    model = setup_minimal_model(X, y, lambda_lin=1e-8, m=2)
    v = np.ones(model.n)
    result = model._woodbury_solve(v)
    
    # Should be finite and not zero
    assert np.all(np.isfinite(result))
    assert not np.allclose(result, 0, atol=1e-6)


def test_woodbury_solve_large_lambda_limit():
    """Test that as λ -> ∞, R^{-1}v → (1/λ)v."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    y = np.array([0.0, 1.0, 0.5])
    
    # Large lambda
    model = setup_minimal_model(X, y, lambda_lin=1e4, m=2)
    v = np.random.default_rng(123).normal(size=model.n)
    result = model._woodbury_solve(v)
    
    # Should approach v/λ, but allow larger tolerance due to conditioning
    expected = v / model.lambda_lin_
    assert_allclose(result, expected, rtol=0.2, atol=1e-8)


def test_woodbury_solve_zero_input():
    """Test that zero input gives zero output."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]])
    y = np.array([0.0, 1.0, 0.5, 0.4])
    model = setup_minimal_model(X, y, lambda_lin=0.1, m=2)
    
    v = np.zeros(model.n)
    result = model._woodbury_solve(v)
    assert_allclose(result, 0, atol=1e-12)


def test_woodbury_solve_dimension_handling():
    """Test proper handling of input vector dimensions."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    y = np.array([0.0, 1.0, 0.5])
    model = setup_minimal_model(X, y, m=2)
    
    # Column vector input
    v_col = np.ones((model.n, 1))
    result_col = model._woodbury_solve(v_col)
    assert result_col.shape == (model.n,)
    
    # Row vector input
    v_row = np.ones((1, model.n))
    result_row = model._woodbury_solve(v_row)
    assert result_row.shape == (model.n,)
    
    # 1D input
    v_1d = np.ones(model.n)
    result_1d = model._woodbury_solve(v_1d)
    assert result_1d.shape == (model.n,)