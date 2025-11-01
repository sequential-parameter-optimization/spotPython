import numpy as np
import pytest
from spotpython.surrogate.kriging import Kriging
import spotpython.surrogate.kriging as kmod  # added for monkeypatch target

@pytest.fixture
def simple_data():
    """Provides a simple dataset for testing."""
    rng = np.random.default_rng(42)
    X = rng.random((10, 2)) * 10
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    return X, y


def test_fit_standard_kriging(simple_data):
    """
    Tests the fit method for a standard Kriging model without approximation.
    """
    X, y = simple_data
    model = Kriging(seed=123)
    
    # Fit the model
    fitted_model = model.fit(X, y)

    # 1. Check if the model returns itself
    assert fitted_model is model

    # 2. Check if internal data matches input
    np.testing.assert_array_equal(model.X_, X)
    np.testing.assert_array_equal(model.y_, y)
    assert model.n == X.shape[0]
    assert model.k == X.shape[1]

    # 3. Check if hyperparameters and model components are set
    assert model.logtheta_loglambda_p_ is not None
    assert model.theta is not None
    assert model.Lambda is not None  # Default is 'regression'
    assert model.negLnLike is not None
    assert model.Psi_ is not None
    assert model.U_ is not None

    # 4. Check shapes of the resulting matrices
    assert model.Psi_.shape == (model.n, model.n)
    assert model.U_.shape == (model.n, model.n)

def test_fit_with_custom_bounds(simple_data):
    """
    Tests fitting with user-provided hyperparameter bounds.
    """
    X, y = simple_data
    # Custom bounds for 2 thetas + 1 lambda
    custom_bounds = [(-1, 1), (-1, 1), (-5, -1)]
    model = Kriging(seed=123)

    # Fit with custom bounds
    model.fit(X, y, bounds=custom_bounds)

    # Check if the optimized parameters are within the custom bounds
    log_thetas = model.logtheta_loglambda_p_[:2]
    log_lambda = model.logtheta_loglambda_p_[2]

    assert np.all(log_thetas >= -1) and np.all(log_thetas <= 1)
    assert -5 <= log_lambda <= -1


def test_fit_interpolation_method(simple_data):
    """
    Tests the 'interpolation' method, which should not optimize Lambda.
    """
    X, y = simple_data
    model = Kriging(method="interpolation", seed=123)
    
    model.fit(X, y)

    # For interpolation, Lambda should be None and not optimized
    assert model.Lambda is None
    # The optimized vector should only contain theta values
    assert len(model.logtheta_loglambda_p_) == model.n_theta


def test_fit_with_optim_p(simple_data):
    """
    Tests fitting with the 'p' parameter optimization enabled.
    """
    X, y = simple_data
    model = Kriging(optim_p=True, seed=123)
    
    model.fit(X, y)

    # Check that p_val is set and is a numpy array
    assert model.p_val is not None
    assert isinstance(model.p_val, np.ndarray)
    # The optimized vector should contain theta, lambda, and p
    assert len(model.logtheta_loglambda_p_) == model.n_theta + 1 + model.n_p

