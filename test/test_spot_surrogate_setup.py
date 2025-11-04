import numpy as np
import pytest

from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, surrogate_control_init, design_control_init

def dummy_fun(X, fun_control=None):
    # Simple sphere function for testing
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)

def test_surrogate_setup_default_kriging():
    fun_control = fun_control_init(lower=np.array([-1, -1]), upper=np.array([1, 1]))
    surrogate_control = surrogate_control_init(method="interpolation")
    spot = Spot(
        fun=dummy_fun,
        fun_control=fun_control,
        surrogate_control=surrogate_control,
    )
    # Should use internal Kriging surrogate
    from spotpython.surrogate.kriging import Kriging
    assert isinstance(spot.surrogate, Kriging)
    assert spot.surrogate.method == "interpolation"
    assert spot.surrogate.var_type == ["num", "num"]

def test_surrogate_setup_custom_surrogate():
    from sklearn.gaussian_process import GaussianProcessRegressor
    fun_control = fun_control_init(lower=np.array([-1, -1]), upper=np.array([1, 1]))
    custom_surrogate = GaussianProcessRegressor()
    spot = Spot(
        fun=dummy_fun,
        fun_control=fun_control,
        surrogate=custom_surrogate,
    )
    # Should use the custom surrogate
    from sklearn.gaussian_process import GaussianProcessRegressor
    assert isinstance(spot.surrogate, GaussianProcessRegressor)

def test_surrogate_setup_preserves_custom_surrogate():
    # If a surrogate is passed, it should not be overwritten
    from sklearn.gaussian_process import GaussianProcessRegressor
    fun_control = fun_control_init(lower=np.array([-1, -1]), upper=np.array([1, 1]))
    custom_surrogate = GaussianProcessRegressor()
    spot = Spot(
        fun=dummy_fun,
        fun_control=fun_control,
        surrogate=custom_surrogate,
    )
    # Call surrogate_setup again with the same surrogate
    spot.surrogate_setup(custom_surrogate)
    assert isinstance(spot.surrogate, GaussianProcessRegressor)

def test_surrogate_setup_sets_attributes():
    fun_control = fun_control_init(lower=np.array([-1, -1]), upper=np.array([1, 1]))
    surrogate_control = surrogate_control_init(method="regression", isotropic=True)
    spot = Spot(
        fun=dummy_fun,
        fun_control=fun_control,
        surrogate_control=surrogate_control,
    )
    # Check that the Kriging surrogate has the correct attributes
    assert hasattr(spot.surrogate, "isotropic")
    assert spot.surrogate.isotropic is True
    assert spot.surrogate.method == "regression"
