import numpy as np
import pytest

from spotpython.surrogate.kriging import Kriging


def _make_xy(k=2):
    X = np.array([[0.0] * k, [0.5] * k, [1.0] * k], dtype=float)
    y = np.array([0.1, 0.2, 0.3], dtype=float)
    return X, y


def test_fit_regression_default_bounds_and_state_updated(monkeypatch):
    X, y = _make_xy(k=2)
    model = Kriging(method="regression")

    seen = {"bounds": None, "x_likelihood": None}

    def fake_max_likelihood(bounds):
        seen["bounds"] = bounds
        # k=2 thetas + 1 lambda
        return np.array([0.1, -0.2, -6.0], dtype=float), -999.0

    def fake_likelihood(x):
        seen["x_likelihood"] = x
        n = X.shape[0]
        return 3.14, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "likelihood", fake_likelihood, raising=True)

    ret = model.fit(X, y)

    assert ret is model

    expected_bounds = [(model.min_theta, model.max_theta)] * X.shape[1] + [
        (model.min_Lambda, model.max_Lambda)
    ]
    assert seen["bounds"] == expected_bounds

    assert np.allclose(model.theta, np.array([0.1, -0.2]))
    assert np.allclose(model.Lambda, np.array([-6.0]))
    assert np.allclose(model.logtheta_loglambda_p_, np.array([0.1, -0.2, -6.0]))
    assert model.negLnLike == pytest.approx(3.14)
    assert np.allclose(model.Psi_, np.eye(X.shape[0]))
    assert np.allclose(model.U_, np.eye(X.shape[0]))
    assert np.allclose(seen["x_likelihood"], np.array([0.1, -0.2, -6.0]))


def test_fit_interpolation_bounds_no_lambda_and_lambda_none(monkeypatch):
    X, y = _make_xy(k=2)
    model = Kriging(method="interpolation")

    seen = {"bounds": None, "x_likelihood": None}

    def fake_max_likelihood(bounds):
        seen["bounds"] = bounds
        # Only k=2 thetas
        return np.array([0.25, -0.75], dtype=float), -1.0

    def fake_likelihood(x):
        seen["x_likelihood"] = x
        n = X.shape[0]
        return 1.23, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "likelihood", fake_likelihood, raising=True)

    model.fit(X, y)

    expected_bounds = [(model.min_theta, model.max_theta)] * X.shape[1]
    assert seen["bounds"] == expected_bounds

    assert np.allclose(model.theta, np.array([0.25, -0.75]))
    assert model.Lambda is None
    assert np.allclose(model.logtheta_loglambda_p_, np.array([0.25, -0.75]))
    assert model.negLnLike == pytest.approx(1.23)
    assert np.allclose(model.Psi_, np.eye(X.shape[0]))
    assert np.allclose(model.U_, np.eye(X.shape[0]))
    assert np.allclose(seen["x_likelihood"], np.array([0.25, -0.75]))


def test_fit_reinterpolation_bounds_and_state_updated(monkeypatch):
    X, y = _make_xy(k=3)
    model = Kriging(method="reinterpolation")

    seen = {"bounds": None}

    def fake_max_likelihood(bounds):
        seen["bounds"] = bounds
        # k=3 thetas + 1 lambda
        return np.array([0.0, 0.1, 0.2, -3.0], dtype=float), -5.0

    def fake_likelihood(x):
        n = X.shape[0]
        return 0.77, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "likelihood", fake_likelihood, raising=True)

    model.fit(X, y)

    expected_bounds = [(model.min_theta, model.max_theta)] * X.shape[1] + [
        (model.min_Lambda, model.max_Lambda)
    ]
    assert seen["bounds"] == expected_bounds

    assert np.allclose(model.theta, np.array([0.0, 0.1, 0.2]))
    assert np.allclose(model.Lambda, np.array([-3.0]))
    assert np.allclose(model.logtheta_loglambda_p_, np.array([0.0, 0.1, 0.2, -3.0]))
    assert model.negLnLike == pytest.approx(0.77)
    assert np.allclose(model.Psi_, np.eye(X.shape[0]))
    assert np.allclose(model.U_, np.eye(X.shape[0]))


def test_fit_isotropic_uses_k_theta_bounds_but_sets_n_theta_1(monkeypatch):
    X, y = _make_xy(k=3)
    model = Kriging(method="regression", isotropic=True)

    seen = {"bounds": None}

    def fake_max_likelihood(bounds):
        seen["bounds"] = bounds
        # despite isotropic, code builds k thetas + 1 lambda bounds
        return np.array([0.5, 0.4, 0.3, -4.0], dtype=float), -2.0

    def fake_likelihood(x):
        n = X.shape[0]
        return 2.22, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "likelihood", fake_likelihood, raising=True)

    model.fit(X, y)

    expected_bounds = [(model.min_theta, model.max_theta)] * X.shape[1] + [
        (model.min_Lambda, model.max_Lambda)
    ]
    assert seen["bounds"] == expected_bounds
    assert model.n_theta == 1
    # Only first theta retained in model.theta for isotropic
    assert np.allclose(model.theta, np.array([0.5]))
    # Lambda is taken from the first param after n_theta
    assert np.allclose(model.Lambda, np.array([0.4]))
    # logtheta_loglambda_p_ still holds all optimized parameters
    assert np.allclose(model.logtheta_loglambda_p_, np.array([0.5, 0.4, 0.3, -4.0]))


def test_fit_respects_explicit_bounds_override(monkeypatch):
    X, y = _make_xy(k=2)
    model = Kriging(method="regression")

    seen = {"bounds": None, "x_likelihood": None}

    def fake_max_likelihood(bounds):
        seen["bounds"] = bounds
        # Must match provided bounds length (2 theta + 1 lambda)
        return np.array([1.0, -1.0, -2.0], dtype=float), -7.0

    def fake_likelihood(x):
        seen["x_likelihood"] = x
        n = X.shape[0]
        return 4.56, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "likelihood", fake_likelihood, raising=True)

    custom_bounds = [(-1.0, 1.0), (-2.0, 2.0), (-8.0, -1.0)]
    model.fit(X, y, bounds=custom_bounds)

    assert seen["bounds"] == custom_bounds
    assert np.allclose(model.theta, np.array([1.0, -1.0]))
    assert np.allclose(model.Lambda, np.array([-2.0]))
    assert model.negLnLike == pytest.approx(4.56)
    assert np.allclose(seen["x_likelihood"], np.array([1.0, -1.0, -2.0]))


def test_fit_regression_with_optim_p_adds_p_bounds_and_sets_p(monkeypatch):
    X, y = _make_xy(k=2)
    model = Kriging(method="regression", optim_p=True, n_p=2, min_p=1.1, max_p=1.9)

    seen = {"bounds": None}

    def fake_max_likelihood(bounds):
        seen["bounds"] = bounds
        # k=2 thetas + 1 lambda + n_p=2 p-values
        return np.array([0.0, 0.2, -5.0, 1.3, 1.7], dtype=float), -10.0

    def fake_likelihood(x):
        n = X.shape[0]
        return 0.5, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "likelihood", fake_likelihood, raising=True)

    model.fit(X, y)

    expected_bounds = (
        [(model.min_theta, model.max_theta)] * X.shape[1]
        + [(model.min_Lambda, model.max_Lambda)]
        + [(model.min_p, model.max_p)] * model.n_p
    )
    assert seen["bounds"] == expected_bounds
    assert np.allclose(model.theta, np.array([0.0, 0.2]))
    assert np.allclose(model.Lambda, np.array([-5.0]))
    assert np.allclose(model.p_val, np.array([1.3, 1.7]))


def test_fit_interpolation_with_optim_p_adds_p_bounds_and_sets_p(monkeypatch):
    X, y = _make_xy(k=2)
    model = Kriging(method="interpolation", optim_p=True, n_p=2, min_p=1.2, max_p=1.8)

    seen = {"bounds": None}

    def fake_max_likelihood(bounds):
        seen["bounds"] = bounds
        # k=2 thetas + n_p=2 p-values
        return np.array([0.3, -0.4, 1.25, 1.55], dtype=float), -3.0

    def fake_likelihood(x):
        n = X.shape[0]
        return 1.0, np.eye(n), np.eye(n)

    monkeypatch.setattr(model, "max_likelihood", fake_max_likelihood, raising=True)
    monkeypatch.setattr(model, "likelihood", fake_likelihood, raising=True)

    model.fit(X, y)

    expected_bounds = (
        [(model.min_theta, model.max_theta)] * X.shape[1]
        + [(model.min_p, model.max_p)] * model.n_p
    )
    assert seen["bounds"] == expected_bounds
    assert model.Lambda is None
    assert np.allclose(model.theta, np.array([0.3, -0.4]))
    assert np.allclose(model.p_val, np.array([1.25, 1.55]))