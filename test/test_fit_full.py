import numpy as np
import pytest
from spotpython.surrogate.kriging import Kriging

def _make_test_data(k=2, n=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, k))
    y = np.sin(X[:, 0]) + (0.1 * X[:, 1] if k > 1 else 0.0)
    return X, y

@pytest.mark.parametrize("use_nystrom", [False, True])
@pytest.mark.parametrize("method", ["regression", "interpolation"])
def test_fit_runs_and_predicts(method, use_nystrom):
    X, y = _make_test_data(k=2, n=10, seed=1)
    model = Kriging(method=method, use_nystrom=use_nystrom, nystrom_m=5)
    model.fit(X, y)
    # Model should store training data
    assert np.allclose(model.X_, X)
    assert np.allclose(model.y_, y)
    # Model should have fitted parameters
    assert model.theta is not None
    # Model should predict something finite at training points
    y_pred = model.predict(X)
    assert np.all(np.isfinite(y_pred))
    assert y_pred.shape == y.shape

def test_fit_with_bounds():
    X, y = _make_test_data(k=2, n=10, seed=2)
    model = Kriging(method="regression")
    bounds = [(-2, 2), (-2, 2), (-6, 0)]
    model.fit(X, y, bounds=bounds)
    assert model.theta.shape == (2,)
    assert model.Lambda.shape == (1,)
    assert np.isfinite(model.negLnLike)

def test_fit_interpolation_exact():
    X, y = _make_test_data(k=1, n=8, seed=3)
    model = Kriging(method="interpolation")
    model.fit(X, y)
    # Should interpolate training data closely
    y_pred = model.predict(X)
    np.testing.assert_allclose(y_pred, y, rtol=1e-2, atol=1e-2)

def test_fit_invalid_input_shape():
    X, y = _make_test_data(k=2, n=10, seed=4)
    model = Kriging()
    # Wrong shape for y
    with pytest.raises(Exception):
        model.fit(X, y.reshape(-1, 1, 1))
    # Wrong shape for X
    with pytest.raises(Exception):
        model.fit(X.reshape(-1, 1, 2), y)

def test_fit_with_optim_p():
    X, y = _make_test_data(k=2, n=10, seed=5)
    model = Kriging(method="regression", optim_p=True, n_p=2)
    model.fit(X, y)
    assert model.p_val is not None
    assert model.p_val.shape == (2,)
    assert np.isfinite(model.negLnLike)