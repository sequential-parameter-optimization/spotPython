import numpy as np
import pytest
from spotpython.surrogate.kriging import Kriging


def _make_model(method: str, k: int = 2) -> Kriging:
    """
    Construct a minimal Kriging instance without calling fit().
    Sets required attributes so likelihood() can run.
    """
    model = Kriging(method=method, n_theta=k)
    # Simple training data (n=3, k features)
    X = np.array([[0.0] * k, [0.5] * k, [1.0] * k], dtype=float)
    y = np.array([0.1, 0.2, 0.3], dtype=float)
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    return model


def _mock_upper(a: float, b: float, c: float, n: int = 3) -> np.ndarray:
    """
    Return an upper triangular matrix with specified off-diagonals:
    [ [0, a, b],
      [0, 0, c],
      [0, 0, 0] ]
    """
    upper = np.zeros((n, n), dtype=float)
    upper[0, 1] = a
    upper[0, 2] = b
    upper[1, 2] = c
    return upper


def _expected_full_from_upper(upper: np.ndarray, lam: float) -> np.ndarray:
    n = upper.shape[0]
    return upper + upper.T + np.eye(n) + np.eye(n) * lam


def test_likelihood_regression_happy_path():
    model = _make_model(method="regression", k=2)
    # Mock correlation upper triangle with small off-diagonals to ensure PD
    a, b, c = 0.2, 0.1, 0.15
    model.build_Psi = lambda: _mock_upper(a, b, c)
    # thetas (unused due to mocked build_Psi) + log10(lambda)
    log_lambda = -6.0  # lambda = 1e-6
    x = np.array([0.0, 0.0, log_lambda], dtype=float)

    neg, Psi, U = model.likelihood(x)

    lam = 10.0**log_lambda
    assert Psi.shape == (3, 3)
    assert np.allclose(Psi, Psi.T)
    # diagonal must be 1 + lambda
    assert np.allclose(np.diag(Psi), (1.0 + lam) * np.ones(3))
    # off-diagonals must match mocked values
    assert Psi[0, 1] == pytest.approx(a)
    assert Psi[0, 2] == pytest.approx(b)
    assert Psi[1, 2] == pytest.approx(c)
    # Cholesky validity
    assert U is not None
    assert np.allclose(Psi, U @ U.T, atol=1e-10)
    # theta slicing propagated
    assert np.allclose(model.theta, x[: model.n_theta])
    # neg log-like should be finite
    assert np.isfinite(neg)

    # Cross-check neg log-like formula using returned U (should match internal computation)
    y = model.y_.flatten()
    n = model.X_.shape[0]
    one = np.ones(n)
    LnDetPsi = 2.0 * np.sum(np.log(np.abs(np.diag(U))))
    temp_y = np.linalg.solve(U, y)
    temp_one = np.linalg.solve(U, one)
    vy = np.linalg.solve(U.T, temp_y)
    vone = np.linalg.solve(U.T, temp_one)
    mu = (one @ vy) / (one @ vone)
    resid = y - one * mu
    tresid = np.linalg.solve(U, resid)
    tresid = np.linalg.solve(U.T, tresid)
    SigmaSqr = (resid @ tresid) / n
    neg_expected = (n / 2.0) * np.log(SigmaSqr) + 0.5 * LnDetPsi
    assert neg == pytest.approx(neg_expected)


def test_likelihood_interpolation_uses_eps_as_lambda():
    model = _make_model(method="interpolation", k=2)
    a, b, c = 0.4, 0.1, 0.25
    model.build_Psi = lambda: _mock_upper(a, b, c)
    x = np.array([0.0, 0.0], dtype=float)  # thetas only

    neg, Psi, U = model.likelihood(x)

    lam = model.eps
    assert Psi.shape == (3, 3)
    assert np.allclose(Psi, Psi.T)
    assert np.allclose(np.diag(Psi), (1.0 + lam) * np.ones(3))
    assert Psi[0, 1] == pytest.approx(a)
    assert Psi[0, 2] == pytest.approx(b)
    assert Psi[1, 2] == pytest.approx(c)
    assert U is not None
    assert np.allclose(Psi, U @ U.T, atol=1e-10)
    assert np.isfinite(neg)


def test_likelihood_reinterpolation_branch_behaves_like_regression():
    model = _make_model(method="reinterpolation", k=2)
    a, b, c = 0.3, 0.15, 0.1
    model.build_Psi = lambda: _mock_upper(a, b, c)
    log_lambda = -3.0  # lambda = 1e-3
    x = np.array([0.0, 0.0, log_lambda], dtype=float)

    neg, Psi, U = model.likelihood(x)

    lam = 10.0**log_lambda
    assert Psi.shape == (3, 3)
    assert np.allclose(Psi, Psi.T)
    assert np.allclose(np.diag(Psi), (1.0 + lam) * np.ones(3))
    assert Psi[0, 1] == pytest.approx(a)
    assert Psi[0, 2] == pytest.approx(b)
    assert Psi[1, 2] == pytest.approx(c)
    assert U is not None
    assert np.allclose(Psi, U @ U.T, atol=1e-10)
    assert np.isfinite(neg)


def test_likelihood_returns_penalty_when_cholesky_fails():
    model = _make_model(method="regression", k=2)
    # Force an indefinite matrix by making off-diagonals bigger than diagonal
    a = b = c = 2.0
    model.build_Psi = lambda: _mock_upper(a, b, c)
    model.penalty = 1234.5
    x = np.array([0.0, 0.0, -6.0], dtype=float)

    neg, Psi, U = model.likelihood(x)

    assert neg == model.penalty
    assert U is None
    lam = 10.0 ** (-6.0)
    expected = _expected_full_from_upper(_mock_upper(a, b, c), lam)
    assert np.allclose(Psi, expected)


def test_likelihood_regression_raises_with_missing_nugget():
    """
    If method is regression but x does not include the nugget (lambda),
    lambda_ becomes an empty array and broadcasting should raise ValueError.
    """
    model = _make_model(method="regression", k=2)
    model.build_Psi = lambda: _mock_upper(0.1, 0.1, 0.1)
    # Missing nugget -> only thetas provided
    x = np.array([0.0, 0.0], dtype=float)

    with pytest.raises(ValueError):
        _ = model.likelihood(x)