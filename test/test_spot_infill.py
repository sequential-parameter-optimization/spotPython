import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init, surrogate_control_init

def quadratic_fun(X, **kwargs):
    return np.sum(X**2, axis=1)

@pytest.fixture
def spot_instance():
    fun_control = fun_control_init(
        lower=np.array([-2, -1]),
        upper=np.array([2, 1]),
        fun_evals=5,
        PREFIX="pytest_infill",
        save_result=False,
        save_experiment=False,
    )
    design_control = design_control_init(init_size=5)
    surrogate_control = surrogate_control_init()
    spot = Spot(fun=quadratic_fun, fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control)
    spot.initialize_design()
    spot.evaluate_initial_design()
    spot.update_stats()
    spot.fit_surrogate()
    return spot

def test_infill_returns_float(spot_instance):
    # Use a point in the middle of the domain
    x = np.array([0.0, 0.0])
    val = spot_instance.infill(x)
    assert isinstance(val, float) or np.isscalar(val) or (isinstance(val, np.ndarray) and val.size == 1)

def test_infill_on_design_points(spot_instance):
    # Should not raise and should return a float for each design point
    for x in spot_instance.X:
        val = spot_instance.infill(x)
        assert isinstance(val, float) or np.isscalar(val) or (isinstance(val, np.ndarray) and val.size == 1)

def test_infill_within_bounds(spot_instance):
    # Try a point at the lower bound
    x = spot_instance.lower.copy()
    val = spot_instance.infill(x)
    assert isinstance(val, float) or np.isscalar(val) or (isinstance(val, np.ndarray) and val.size == 1)
    # Try a point at the upper bound
    x = spot_instance.upper.copy()
    val = spot_instance.infill(x)
    assert isinstance(val, float) or np.isscalar(val) or (isinstance(val, np.ndarray) and val.size == 1)

def test_infill_shape(spot_instance):
    # Should accept 1D input of correct length
    x = np.zeros(spot_instance.k)
    val = spot_instance.infill(x)
    assert np.isscalar(val) or (isinstance(val, np.ndarray) and val.shape == (1,))