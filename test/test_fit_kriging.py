import numpy as np
import pytest

from spotpython.surrogate.kriging import Kriging


def _make_xy(k=2):
    X = np.array([[0.0] * k, [0.5] * k, [1.0] * k], dtype=float)
    y = np.array([0.1, 0.2, 0.3], dtype=float)
    return X, y


def test_fit_calls__likelihood_exact_not_likelihood(monkeypatch):
    X, y = _make_xy(k=2)
    model = Kriging(method="regression")

    seen = {"max_bounds": None, "called_like": False, "called_exact": False}

    def fake_max_likelihood(bounds):
        seen["max_bounds"] = bounds
        return np.array([0.0, 0.0, -6.0], dtype=float), -123.0

    def fake_likelihood(x):
        seen["called_like"] = True
        n = X.shape[0]
        return 3.14, np.eye(n), np.eye(n)

    def fake_likelihood_exact(x):
        seen["called_exact"] = True
        n = X.shape[0]
        Psi = 2.0 * np.eye(n)
        U = np.sqrt(2.0) * np.eye(n)
        return 9.87, Psi, U

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "likelihood", fake_likelihood, raising=True)
    monkeypatch.setattr(model, "_likelihood_exact", fake_likelihood_exact, raising=True)

    model.fit(X, y)

    # Ensure exact path was used
    assert seen["called_exact"] is True
    assert seen["called_like"] is False

    # Final state comes from _likelihood_exact
    n = X.shape[0]
    assert model.negLnLike == pytest.approx(9.87)
    assert np.allclose(model.Psi_, 2.0 * np.eye(n))
    assert np.allclose(model.U_, np.sqrt(2.0) * np.eye(n))


def test_fit_sets_training_state_and_minmax(monkeypatch):
    X, y = _make_xy(k=3)
    model = Kriging(method="interpolation")

    def fake_max_likelihood(bounds):
        # only k=3 thetas for interpolation
        return np.array([0.1, -0.2, 0.3], dtype=float), -1.0

    def fake_likelihood_exact(x):
        n = X.shape[0]
        return 1.11, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "_likelihood_exact", fake_likelihood_exact, raising=True)

    model.fit(X, y)

    # X_, y_, n, k
    assert np.allclose(model.X_, X)
    assert np.allclose(model.y_, y)
    assert model.n == X.shape[0]
    assert model.k == X.shape[1]

    # min_X / max_X computed
    assert np.allclose(model.min_X, np.min(X, axis=0))
    assert np.allclose(model.max_X, np.max(X, axis=0))

    # Theta set from solution vector
    assert np.allclose(model.theta, np.array([0.1, -0.2, 0.3]))
    # Lambda remains None for interpolation
    assert model.Lambda is None


def test_fit_updates_log_once(monkeypatch):
    X, y = _make_xy(k=2)
    model = Kriging(method="regression")

    calls = {"update": 0}

    def fake_max_likelihood(bounds):
        return np.array([0.0, 0.0, -6.0], dtype=float), -2.2

    def fake_likelihood_exact(x):
        n = X.shape[0]
        return 0.33, np.eye(n), np.eye(n)

    def fake_update_log():
        calls["update"] += 1

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "_likelihood_exact", fake_likelihood_exact, raising=True)
    monkeypatch.setattr(model, "_update_log", fake_update_log, raising=True)

    ret = model.fit(X, y)

    assert ret is model
    assert calls["update"] == 1


def test_fit_sets_logtheta_vector_and_shapes(monkeypatch):
    X, y = _make_xy(k=2)
    model = Kriging(method="regression")

    sol = np.array([0.4, -0.7, -5.0], dtype=float)

    def fake_max_likelihood(bounds):
        return sol.copy(), -999.0

    def fake_likelihood_exact(x):
        n = X.shape[0]
        return 2.5, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "_likelihood_exact", fake_likelihood_exact, raising=True)

    model.fit(X, y)

    # Stored full parameter vector
    assert np.allclose(model.logtheta_loglambda_p_, sol)
    # Theta and Lambda slices
    assert np.allclose(model.theta, sol[:2])
    assert np.allclose(model.Lambda, np.array([sol[2]]))

    # Psi_ and U_ are n x n
    n = X.shape[0]
    assert model.Psi_.shape == (n, n)
    assert model.U_.shape == (n, n)


def test_fit_returns_self_and_keeps_method(monkeypatch):
    X, y = _make_xy(k=2)
    model = Kriging(method="reinterpolation")

    def fake_max_likelihood(bounds):
        # k=2 thetas + 1 lambda
        return np.array([0.0, 0.1, -3.0], dtype=float), -4.2

    def fake_likelihood_exact(x):
        n = X.shape[0]
        return 7.7, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "_likelihood_exact", fake_likelihood_exact, raising=True)

    ret = model.fit(X, y)

    assert ret is model
    assert model.method == "reinterpolation"
    # basic state
    assert np.allclose(model.theta, np.array([0.0, 0.1]))
    assert np.allclose(model.Lambda, np.array([-3.0]))