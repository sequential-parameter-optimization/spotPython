import os
import sys
import numpy as np
import pytest

# Ensure src/ is on sys.path when running tests from repo root with src-layout
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from spotpython.surrogate.kriging import Kriging


def _make_model(var_type, X, y, isotropic=False, theta_log=None):
    k = X.shape[1]
    model = Kriging(method="regression", n_theta=(1 if isotropic else k), var_type=var_type, isotropic=isotropic)
    model.X_ = np.asarray(X, dtype=float)
    model.y_ = np.asarray(y, dtype=float)
    model.n, model.k = model.X_.shape
    model._set_variable_types()
    # Set log10-theta(s)
    if theta_log is None:
        theta_log = np.zeros(1 if isotropic else k, dtype=float)
    model.theta = np.asarray(theta_log, dtype=float)
    return model


def _weighted_sqeuclidean(u, V, w):
    # u: (k,), V: (n, k), w: (k,)
    return ((w * (V - u) ** 2).sum(axis=1)).astype(float)


def test_build_psi_vec_numeric_anisotropic():
    # Two numeric variables with different weights
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]], dtype=float)
    y = np.array([0.1, 0.2, 0.3], dtype=float)
    var_type = ["num", "num"]
    # theta (log10): [0, 1] -> weights w = [1, 10]
    theta_log = np.array([0.0, 1.0], dtype=float)
    model = _make_model(var_type, X, y, isotropic=False, theta_log=theta_log)

    x = np.array([0.5, 0.25], dtype=float)
    psi = model.build_psi_vec(x)

    w = 10.0 ** theta_log  # [1, 10]
    D_expected = _weighted_sqeuclidean(x, X, w)
    psi_expected = np.exp(-D_expected)

    assert psi.shape == (X.shape[0],)
    assert np.allclose(psi, psi_expected, rtol=1e-12, atol=1e-12)
    assert np.all((psi > 0) & (psi <= 1))


def test_build_psi_vec_isotropic_numeric():
    # Isotropic: single theta applied to both dimensions
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]], dtype=float)
    y = np.array([0.1, 0.2, 0.3], dtype=float)
    var_type = ["num", "num"]
    theta_log = np.array([0.5], dtype=float)  # weight = 10**0.5
    model = _make_model(var_type, X, y, isotropic=True, theta_log=theta_log)

    x = np.array([0.2, 0.7], dtype=float)
    w_scalar = float(10.0 ** theta_log[0])
    w = np.array([w_scalar, w_scalar], dtype=float)
    D_expected = _weighted_sqeuclidean(x, X, w)
    psi_expected = np.exp(-D_expected)

    psi = model.build_psi_vec(x)
    assert psi.shape == (X.shape[0],)
    assert np.allclose(psi, psi_expected, rtol=1e-12, atol=1e-12)


def test_build_psi_vec_mixed_ordered_and_factor_with_sqeuclidean_metric():
    # One ordered and one factor-like column; force factor metric to 'sqeuclidean' to accept weights
    X = np.array([[0.0, 0.0],
                  [0.5, 1.0],
                  [1.0, 2.0]], dtype=float)
    y = np.array([0.0, 0.0, 0.0], dtype=float)
    var_type = ["num", "factor"]  # first ordered, second treated as factor
    theta_log = np.array([0.0, 1.0], dtype=float)  # weights [1, 10]
    model = _make_model(var_type, X, y, isotropic=False, theta_log=theta_log)
    # Use sqeuclidean for factor branch so cdist accepts `w`
    model.metric_factorial = "sqeuclidean"

    x = np.array([0.25, 1.0], dtype=float)
    w = 10.0 ** theta_log  # [1, 10]

    # Ordered contribution uses dim 0 with weight 1
    D_ordered = w[0] * (X[:, 0] - x[0]) ** 2
    # Factor contribution uses dim 1 with weight 10
    D_factor = w[1] * (X[:, 1] - x[1]) ** 2
    D_expected = D_ordered + D_factor
    psi_expected = np.exp(-D_expected)

    psi = model.build_psi_vec(x)
    assert psi.shape == (X.shape[0],)
    assert np.allclose(psi, psi_expected, rtol=1e-12, atol=1e-12)


def test_build_psi_vec_returns_one_on_exact_match():
    # If x equals a training point, corresponding psi element should be 1.0
    X = np.array([[0.0, 0.0],
                  [0.5, 0.5],
                  [1.0, 1.0]], dtype=float)
    y = np.array([0.1, 0.2, 0.3], dtype=float)
    var_type = ["num", "num"]
    theta_log = np.array([0.0, 0.0], dtype=float)  # weights [1, 1]
    model = _make_model(var_type, X, y, isotropic=False, theta_log=theta_log)

    x = X[1].copy()
    psi = model.build_psi_vec(x)

    assert psi[1] == pytest.approx(1.0)
    assert np.all(psi <= 1.0 + 1e-15)