import numpy as np
import pytest
from spotpython.surrogate.kriging import Kriging

def _make_test_data(k=2, n=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, k))
    y = np.sin(X[:, 0]) + (0.1 * X[:, 1] if k > 1 else 0.0)
    return X, y

@pytest.mark.parametrize("method", ["regression", "interpolation"])
def test_pred_exact_output_shape_and_type(method):
    X, y = _make_test_data(k=2, n=12, seed=1)
    model = Kriging(method=method)
    model.fit(X, y)
    x0 = X[0]
    mean, std, ei = model._pred_exact(x0)
    assert np.isscalar(mean)
    assert np.isscalar(std)
    assert ei is None or np.isscalar(ei)

def test_pred_exact_matches_predict():
    X, y = _make_test_data(k=2, n=15, seed=2)
    model = Kriging(method="regression")
    model.fit(X, y)
    x0 = X[3]
    mean1, std1, _ = model._pred_exact(x0)
    mean2 = model.predict(x0.reshape(1, -1))[0]
    assert np.allclose(mean1, mean2, rtol=1e-6)

def test_pred_exact_batch_consistency():
    X, y = _make_test_data(k=2, n=10, seed=3)
    model = Kriging(method="regression")
    model.fit(X, y)
    means = []
    for x in X:
        mean, std, _ = model._pred_exact(x)
        means.append(mean)
    means = np.array(means)
    means_predict = model.predict(X)
    np.testing.assert_allclose(means, means_predict, rtol=1e-6)

def test_pred_exact_with_ei():
    X, y = _make_test_data(k=2, n=10, seed=4)
    model = Kriging(method="regression")
    model.fit(X, y)
    model.return_ei = True
    x0 = X[0]
    mean, std, ei = model._pred_exact(x0)
    assert np.isscalar(mean)
    assert np.isscalar(std)
    assert np.isscalar(ei)

def test_pred_exact_invalid_input():
    X, y = _make_test_data(k=2, n=10, seed=5)
    model = Kriging(method="regression")
    model.fit(X, y)
    # Wrong dimension
    with pytest.raises(Exception):
        model._pred_exact(np.array([1.0, 2.0, 3.0]))