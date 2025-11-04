import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

# Use a top-level function for pickling safety
def quadratic_fun(X, **kwargs):
    return np.sum(X**2, axis=1)

@pytest.fixture
def spot_instance():
    fun_control = fun_control_init(
        lower=np.array([-2, -1]),
        upper=np.array([2, 1]),
        fun_evals=5,
        PREFIX="pytest_generate_random_point",
        save_result=False,
        save_experiment=False,
    )
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=quadratic_fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design()
    spot.evaluate_initial_design()
    spot.update_stats()
    spot.fit_surrogate()
    return spot

def test_generate_random_point_shape_and_bounds(spot_instance):
    X0, y0 = spot_instance.generate_random_point()
    # X0 should have shape (1, k)
    assert X0.shape[1] == spot_instance.k
    # y0 should have shape (1,) or (1, 1)
    assert y0.shape[0] == X0.shape[0]
    # X0 values should be within bounds
    assert np.all(X0 >= spot_instance.lower)
    assert np.all(X0 <= spot_instance.upper)

def test_generate_random_point_y_is_float(spot_instance):
    X0, y0 = spot_instance.generate_random_point()
    # y0 should be a float or array of floats
    assert np.issubdtype(y0.dtype, np.floating)

def test_generate_random_point_no_nan(spot_instance):
    X0, y0 = spot_instance.generate_random_point()
    assert not np.isnan(X0).any()
    assert not np.isnan(y0).any()

def test_generate_random_point_multiple_calls_unique_seed():
    points = set()
    for i in range(5):
        fun_control = fun_control_init(
            lower=np.array([-2, -1]),
            upper=np.array([2, 1]),
            fun_evals=5,
            PREFIX="pytest_generate_random_point",
            save_result=False,
            save_experiment=False,
            seed=123 + i,  # different seed each time
        )
        design_control = design_control_init(init_size=3)
        spot = Spot(fun=quadratic_fun, fun_control=fun_control, design_control=design_control)
        spot.initialize_design()
        spot.evaluate_initial_design()
        spot.update_stats()
        spot.fit_surrogate()
        X0, _ = spot.generate_random_point()
        points.add(tuple(X0.flatten()))
    assert len(points) > 1