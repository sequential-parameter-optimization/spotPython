import pytest
import numpy as np
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import spot
from spotpython.utils.init import (
    fun_control_init,
    surrogate_control_init,
    design_control_init
)

@pytest.fixture
def setup_spot_with_anisotropic():
    """Fixture to create a Spot instance with anisotropic theta"""
    fun = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1, -1]),
        upper=np.array([1, 1, 1]),
        fun_evals=20,
        var_name=["x1", "x2", "x3"],
    )
    design_control = design_control_init(init_size=10)
    surrogate_control = surrogate_control_init(
        method="interpolation"
    )
    S = spot.Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control
    )
    S.run()
    return S

@pytest.fixture
def setup_spot_with_single_theta():
    """Fixture to create a Spot instance with single theta"""
    fun = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1, -1]),
        upper=np.array([1, 1, 1]),
        fun_evals=20,
        var_name=["x1", "x2", "x3"],
    )
    design_control = design_control_init(init_size=10)
    surrogate_control = surrogate_control_init(
        isotropic=True,
        method="interpolation"
    )
    S = spot.Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control
    )
    S.run()
    return S

def test_importance_with_anisotropic_theta(setup_spot_with_anisotropic):
    """Test importance calculation with anisotropic theta"""
    S = setup_spot_with_anisotropic
    importance = S.get_importance()
    
    # Check if importance is returned as a list
    assert isinstance(importance, list)
    
    # Check if importance has correct length
    assert len(importance) == len(S.all_var_name)
    
    # Check if importance values are between 0 and 100
    assert all(0 <= imp <= 100 for imp in importance)
    
    # Check if at least one importance value is 100
    assert max(importance) == 100

def test_importance_with_single_theta(setup_spot_with_single_theta):
    """Test importance calculation with single theta"""
    S = setup_spot_with_single_theta
    importance = S.get_importance()
    
    # Check if importance is returned as a list
    assert isinstance(importance, list)
    
    # Check if importance has correct length
    assert len(importance) == 0
    

def test_importance_without_surrogate():
    """Test get_importance when no surrogate is available"""
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )
    S = spot.Spot(fun=lambda x: x, fun_control=fun_control, surrogate=None)
    importance = S.get_importance()
    assert importance == []

def test_importance_without_theta_attribute():
    """Test get_importance when surrogate has no theta attribute"""
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )
    S = spot.Spot(fun=lambda x: x, fun_control=fun_control)
    class DummySurrogate:
        def __init__(self):
            self.n_theta = 2
    S.surrogate = DummySurrogate()
    importance = S.get_importance()
    assert importance == []

def test_importance_without_all_var_name():
    """Test get_importance when all_var_name is not available"""