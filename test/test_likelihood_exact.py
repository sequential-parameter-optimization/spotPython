import numpy as np
import pytest

from spotpython.surrogate.kriging import Kriging


def setup_minimal_model(X, y, *, method="regression", var_type=None, isotropic=False):
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, k = X.shape
    model = Kriging(method=method, var_type=(var_type or ["num"] * k), isotropic=isotropic)
    # Minimal internal state required by _likelihood_exact/build_Psi
    model.X_ = X
    model.y_ = y
    model.n, model.k = n, k
    model._set_variable_types()
    model._set_theta()
    return model


def manual_negloglike_from_U(U, y):
    n = y.size
    one = np.ones(n)
    # Ln |Psi|
    LnDetPsi = 2.0 * np.sum(np.log(np.abs(np.diag(U))))
    # Solve with U (lower-triangular)
    temp_y = np.linalg.solve(U, y)
    temp_one = np.linalg.solve(U, one)
    vy = np.linalg.solve(U.T, temp_y)
    vone = np.linalg.solve(U.T, temp_one)
    mu = (one @ vy) / (one @ vone)
    resid = y - one * mu
    tresid = np.linalg.solve(U, resid)
    tresid = np.linalg.solve(U.T, tresid)
    sigma2 = (resid @ tresid) / n
    return (n / 2.0) * np.log(sigma2) + 0.5 * LnDetPsi


def test_likelihood_exact_regression_anisotropic_shapes_and_chol():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(15, 2))
    y = np.sin(X[:, 0]) + 0.2 * X[:, 1]

    model = setup_minimal_model(X, y, method="regression", isotropic=False)
    # x = [log10(theta1), log10(theta2), log10(lambda)]
    x = np.array([0.0, 0.0, -3.0])  # theta=1, lambda=1e-3

    neg, Psi, U = model._likelihood_exact(x)

    assert np.isfinite(neg)
    assert Psi.shape == (X.shape[0], X.shape[0])
    assert U.shape == Psi.shape
    # Psi must be symmetric with diagonal 1 + lambda
    assert np.allclose(Psi, Psi.T, atol=1e-12)
    lam = 10.0 ** x[2]
    assert np.allclose(np.diag(Psi), 1.0 + lam, atol=1e-12)
    # U should be a valid Cholesky of Psi
    assert np.allclose(U @ U.T, Psi, rtol=1e-9, atol=1e-12)
    # Neg log-likelihood must match manual computation
    neg_manual = manual_negloglike_from_U(U, y)
    assert np.isfinite(neg_manual)
    assert np.allclose(neg, neg_manual, rtol=1e-9, atol=1e-10)


def test_likelihood_exact_isotropic_vs_anisotropic_when_same_theta():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(12, 3))
    y = np.cos(X[:, 0]) + 0.1 * X[:, 1] - 0.05 * X[:, 2]

    # Anisotropic with equal per-dim log10(theta)=0.3
    model_aniso = setup_minimal_model(X, y, method="regression", isotropic=False)
    x_aniso = np.array([0.3, 0.3, 0.3, -2.0])  # 3 thetas + lambda
    neg_a, Psi_a, U_a = model_aniso._likelihood_exact(x_aniso)

    # Isotropic with single theta 0.3
    model_iso = setup_minimal_model(X, y, method="regression", isotropic=True)
    x_iso = np.array([0.3, -2.0])  # 1 theta + lambda
    neg_i, Psi_i, U_i = model_iso._likelihood_exact(x_iso)

    assert np.allclose(Psi_a, Psi_i, atol=1e-12)
    assert np.allclose(U_a @ U_a.T, U_i @ U_i.T, rtol=1e-9, atol=1e-12)
    assert np.allclose(neg_a, neg_i, rtol=1e-9, atol=1e-10)


def test_likelihood_exact_interpolation_matches_regression_with_lambda_eq_eps():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)

    # Interpolation (lambda taken as eps)
    model_int = setup_minimal_model(X, y, method="interpolation", isotropic=False)
    x_int = np.array([0.0, 0.0])  # just thetas
    neg_int, Psi_int, U_int = model_int._likelihood_exact(x_int)

    # Regression with lambda set to eps (on log10 scale)
    model_reg = setup_minimal_model(X, y, method="regression", isotropic=False)
    log10_eps = np.log10(model_reg.eps)
    x_reg = np.array([0.0, 0.0, log10_eps])
    neg_reg, Psi_reg, U_reg = model_reg._likelihood_exact(x_reg)

    # Expect close equality
    assert np.allclose(Psi_int, Psi_reg, rtol=1e-12, atol=1e-12)
    assert np.allclose(U_int @ U_int.T, U_reg @ U_reg.T, rtol=1e-9, atol=1e-12)
    assert np.allclose(neg_int, neg_reg, rtol=1e-9, atol=1e-10)


def test_likelihood_exact_penalty_on_cholesky_failure(monkeypatch):
    # Force cholesky to raise LinAlgError to test penalty branch
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(8, 2))
    y = np.sin(X[:, 0])
    model = setup_minimal_model(X, y, method="regression", isotropic=False)
    x = np.array([0.0, 0.0, -3.0])

    def raise_chol(_):
        from numpy.linalg import LinAlgError as LAErr
        raise LAErr("forced failure")

    monkeypatch.setattr(np.linalg, "cholesky", raise_chol)
    neg, Psi, U = model._likelihood_exact(x)

    assert neg == model.penalty
    assert Psi.shape == (X.shape[0], X.shape[0])
    assert U is None