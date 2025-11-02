import numpy as np
import pytest
from spotpython.surrogate.kriging import Kriging

@pytest.fixture
def simple_1d_data():
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel()
    return X, y

@pytest.fixture
def simple_2d_data():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    return X, y

def test_predict_1d_exact(simple_1d_data):
    X, y = simple_1d_data
    model = Kriging(method="regression")
    model.fit(X, y)
    X_test = np.linspace(0, 1, 10).reshape(-1, 1)
    y_pred = model.predict(X_test)
    assert y_pred.shape == (10,)
    assert np.all(np.isfinite(y_pred))

def test_predict_1d_nystrom(simple_1d_data):
    X, y = simple_1d_data
    model = Kriging(method="regression", use_nystrom=True, nystrom_m=3)
    model.fit(X, y)
    X_test = np.linspace(0, 1, 10).reshape(-1, 1)
    y_pred = model.predict(X_test)
    assert y_pred.shape == (10,)
    assert np.all(np.isfinite(y_pred))

def test_predict_2d_exact(simple_2d_data):
    X, y = simple_2d_data
    model = Kriging(method="regression")
    model.fit(X, y)
    X_test = np.array([[0.5, 0.5], [0.2, 0.8]])
    y_pred = model.predict(X_test)
    assert y_pred.shape == (2,)
    assert np.all(np.isfinite(y_pred))

def test_predict_2d_nystrom(simple_2d_data):
    X, y = simple_2d_data
    model = Kriging(method="regression", use_nystrom=True, nystrom_m=2)
    model.fit(X, y)
    X_test = np.array([[0.5, 0.5], [0.2, 0.8]])
    y_pred = model.predict(X_test)
    assert y_pred.shape == (2,)
    assert np.all(np.isfinite(y_pred))

def test_predict_return_std(simple_1d_data):
    X, y = simple_1d_data
    model = Kriging(method="regression")
    model.fit(X, y)
    X_test = np.linspace(0, 1, 4).reshape(-1, 1)
    y_pred, y_std = model.predict(X_test, return_std=True)
    assert y_pred.shape == (4,)
    assert y_std.shape == (4,)
    assert np.all(y_std >= 0)

def test_predict_return_val_all(simple_1d_data):
    X, y = simple_1d_data
    model = Kriging(method="regression")
    model.fit(X, y)
    X_test = np.linspace(0, 1, 3).reshape(-1, 1)
    y_pred, y_std, y_ei = model.predict(X_test, return_val="all")
    assert y_pred.shape == (3,)
    assert y_std.shape == (3,)
    assert y_ei.shape == (3,)

def test_predict_return_val_ei(simple_1d_data):
    X, y = simple_1d_data
    model = Kriging(method="regression")
    model.fit(X, y)
    X_test = np.linspace(0, 1, 3).reshape(-1, 1)
    y_ei = model.predict(X_test, return_val="ei")
    assert y_ei.shape == (3,)
    assert np.all(np.isfinite(y_ei))