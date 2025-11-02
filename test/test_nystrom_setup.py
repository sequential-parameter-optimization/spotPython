import numpy as np
import pytest

from spotpython.surrogate.kriging import Kriging


def _make_xy(k=2, n=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, k))
    y = np.sin(X[:, 0]) + (0.1 if k > 1 else 0.0) * (X[:, 1] if k > 1 else 0.0)
    return X, np.asarray(y, dtype=float)


def test_fit_nystrom_isotropic_bounds_match_implementation(monkeypatch):
    # Current implementation builds bounds with k theta entries even if isotropic=True
    k = 4
    X, y = _make_xy(k=k, n=16, seed=11)
    model = Kriging(method="regression", use_nystrom=True, isotropic=True)

    seen = {"bounds": None}

    def fake_max_likelihood(bounds):
        seen["bounds"] = bounds
        # 1 theta logically, but implementation passes k thetas + 1 lambda
        return np.array([0.25] * k + [-4.0], dtype=float), -2.2

    def fake_like_nystrom(x):
        return 1.5, None, None

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "_likelihood_nystrom", fake_like_nystrom, raising=True)

    model.fit(X, y)

    # Expect k theta bounds + 1 lambda bound
    expected_bounds = [(model.min_theta, model.max_theta)] * k + [(model.min_Lambda, model.max_Lambda)]
    assert seen["bounds"] == expected_bounds
    # Theta got set from solution vector
    assert model.theta.shape[0] in (1, k)  # accept either internal collapse or keep-k
    assert np.isfinite(model.negLnLike)


def test_fit_dispatch_exact_when_nystrom_disabled(monkeypatch):
    X, y = _make_xy(k=3, n=20, seed=5)
    model = Kriging(method="regression", use_nystrom=False)

    called = {"exact": False, "nystrom": False}

    def fake_max_likelihood(bounds):
        # k=3 theta + 1 lambda
        return np.array([0.0, -0.1, 0.2, -3.0], dtype=float), -9.0

    def fake_like_exact(x):
        called["exact"] = True
        n = X.shape[0]
        return 2.0, np.eye(n), np.eye(n)

    def fake_like_nystrom(x):
        called["nystrom"] = True
        return 99.0, None, None

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "_likelihood_exact", fake_like_exact, raising=True)
    monkeypatch.setattr(model, "_likelihood_nystrom", fake_like_nystrom, raising=True)

    model.fit(X, y)

    assert called["exact"] is True
    assert called["nystrom"] is False
    assert np.allclose(model.theta, np.array([0.0, -0.1, 0.2]))
    assert np.allclose(model.Lambda, np.array([-3.0]))
    assert model.Psi_ is not None and model.U_ is not None


def test_fit_nystrom_sets_lambda_lin_from_Lambda(monkeypatch):
    X, y = _make_xy(k=2, n=14, seed=7)
    model = Kriging(method="regression", use_nystrom=True)

    def fake_max_likelihood(bounds):
        # log10(lambda) = -2.5 -> lambda_lin ≈ 10**-2.5
        return np.array([0.1, -0.2, -2.5], dtype=float), -1.0

    def fake_like_nystrom(x):
        return 3.3, None, None

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "_likelihood_nystrom", fake_like_nystrom, raising=True)

    model.fit(X, y)

    assert np.allclose(model.theta, np.array([0.1, -0.2]))
    assert np.allclose(model.Lambda, np.array([-2.5]))
    # lambda_lin_ computed from Lambda after fit
    assert np.isclose(model.lambda_lin_, 10.0 ** (-2.5), rtol=1e-12, atol=0.0)


def test_nystrom_setup_repeatable_with_same_seed_on_same_model():
    X, y = _make_xy(k=2, n=18, seed=3)
    model = Kriging(method="regression", use_nystrom=True)
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    model.n_theta = model.k
    model.theta = np.array([-1.0, -1.0])
    model.lambda_lin_ = 1e-2
    model.nystrom_m = 7
    model.nystrom_seed = 123

    # Call twice; since RNG is re-seeded each call, indices should be identical
    model._nystrom_setup()
    idx1 = model.landmark_idx_.copy()
    C1 = model.C_.copy()
    model._nystrom_setup()
    idx2 = model.landmark_idx_.copy()
    C2 = model.C_.copy()

    assert np.array_equal(idx1, idx2)
    assert np.allclose(C1, C2)


def test_woodbury_solve_zero_vector_returns_zero():
    X, y = _make_xy(k=2, n=12, seed=9)
    model = Kriging(method="regression", use_nystrom=True)
    # Minimal training state
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    model.n_theta = model.k
    model.theta = np.array([-1.2, -1.0])
    model.lambda_lin_ = 1e-2
    model.nystrom_m = 5
    model.nystrom_seed = 8
    model._nystrom_setup()

    # Precompute M_chol
    C = model.C_
    W_chol = model.W_chol_
    W = W_chol @ W_chol.T
    CtC = C.T @ C
    M = W + (CtC / model.lambda_lin_)
    try:
        model.M_chol_ = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        model.M_chol_ = np.linalg.cholesky(M + 1e-10 * np.eye(M.shape[0]))

    v = np.zeros(model.n)
    out = model._woodbury_solve(v)
    assert np.allclose(out, np.zeros_like(v))


def test_nystrom_setup_handles_single_sample_n_equals_one():
    X = np.array([[0.25, -0.5]], dtype=float)  # n=1, k=2
    y = np.array([0.1], dtype=float)
    model = Kriging(method="regression", use_nystrom=True)
    model.X_ = X
    model.y_ = y
    model.n, model.k = X.shape
    model._set_variable_types()
    model.n_theta = model.k
    model.theta = np.array([-1.0, -1.0])
    model.lambda_lin_ = 1e-2
    model.nystrom_m = None  # default path should clamp to 1 for n==1
    model.nystrom_seed = 0

    model._nystrom_setup()

    # With n=1, m=1
    assert model.C_.shape == (1, 1)
    assert model.W_chol_.shape == (1, 1)
    # W has unit diagonal
    W = model.W_chol_ @ model.W_chol_.T
    assert np.allclose(W[0, 0], 1.0, atol=1e-12)