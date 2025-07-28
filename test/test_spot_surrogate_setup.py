import pytest
import numpy as np
from spotpython.spot import spot
from spotpython.utils.init import (
    fun_control_init,
    surrogate_control_init
)
from sklearn.gaussian_process import GaussianProcessRegressor
from spotpython.fun.objectivefunctions import Analytical

@pytest.fixture
def basic_spot_instance():
    """Creates a basic Spot instance with minimal configuration."""
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )
    return spot.Spot(
        fun=Analytical().fun_sphere,
        fun_control=fun_control
    )

def test_surrogate_setup_with_custom_surrogate(basic_spot_instance):
    """Test surrogate setup with a custom surrogate model."""
    # Create custom surrogate
    custom_surrogate = GaussianProcessRegressor()
    
    # Setup surrogate
    basic_spot_instance.surrogate_setup(custom_surrogate)
    
    # Check if surrogate is set correctly
    assert basic_spot_instance.surrogate == custom_surrogate
    assert isinstance(basic_spot_instance.surrogate, GaussianProcessRegressor)

def test_surrogate_setup_with_default_kriging():
    """Test surrogate setup with default Kriging model."""
    # Create spot instance with specific surrogate control
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )
    surrogate_control = surrogate_control_init(
        method="interpolation",
        n_theta="anisotropic"
    )
    S = spot.Spot(
        fun=Analytical().fun_sphere,
        fun_control=fun_control,
        surrogate_control=surrogate_control
    )
    
    # Setup surrogate
    S.surrogate_setup(None)
    
    # Check if Kriging surrogate is created with correct parameters
    assert S.surrogate is not None
    assert S.surrogate.method == "interpolation"
    assert S.surrogate.n_theta == 2  # anisotropic means n_theta equals dimension

def test_surrogate_setup_parameters():
    """Test if surrogate setup correctly uses parameters from surrogate_control."""
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )
    surrogate_control = surrogate_control_init(
        method="interpolation",
        n_theta=1,
        min_theta=-2,
        max_theta=2,
        theta_init_zero=True,
        log_level=50
    )
    S = spot.Spot(
        fun=Analytical().fun_sphere,
        fun_control=fun_control,
        surrogate_control=surrogate_control
    )
    
    # Setup surrogate
    S.surrogate_setup(None)
    
    # Check if parameters are set correctly
    assert S.surrogate.method == "interpolation"
    assert S.surrogate.n_theta == 1
    assert S.surrogate.min_theta == -2
    assert S.surrogate.max_theta == 2
    assert S.surrogate.theta_init_zero == True
    assert S.surrogate.log_level == 50

def test_surrogate_setup_var_types():
    """Test if surrogate setup handles variable types correctly."""
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        var_type=["num", "factor"]
    )
    surrogate_control = surrogate_control_init(
        method="interpolation"
    )
    S = spot.Spot(
        fun=Analytical().fun_sphere,
        fun_control=fun_control,
        surrogate_control=surrogate_control
    )
    
    # Setup surrogate
    S.surrogate_setup(None)
    
    # Check if variable types are set correctly
    assert S.surrogate.var_type == ["num", "factor"]


def test_surrogate_setup_with_none(basic_spot_instance):
    """Test if surrogate setup handles None input correctly."""
    # Setup surrogate with None
    basic_spot_instance.surrogate_setup(None)
    
    # Check if default Kriging surrogate is created
    assert basic_spot_instance.surrogate is not None
    assert hasattr(basic_spot_instance.surrogate, 'predict')