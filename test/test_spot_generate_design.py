import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def quadratic_fun(X, **kwargs):
    return np.sum(X**2, axis=1)

@pytest.fixture
def spot_instance():
    fun_control = fun_control_init(
        lower=np.array([-2, -1]),
        upper=np.array([2, 1]),
        fun_evals=5,
        PREFIX="pytest_generate_design",
        save_result=False,
        save_experiment=False,
    )
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=quadratic_fun, fun_control=fun_control, design_control=design_control)
    return spot

def test_generate_design_shape(spot_instance):
    n_points = 7
    repeats = 2
    X = spot_instance.generate_design(size=n_points, repeats=repeats, lower=spot_instance.lower, upper=spot_instance.upper)
    assert X.shape[0] == n_points * repeats
    assert X.shape[1] == spot_instance.k

def test_generate_design_within_bounds(spot_instance):
    n_points = 5
    repeats = 1
    X = spot_instance.generate_design(size=n_points, repeats=repeats, lower=spot_instance.lower, upper=spot_instance.upper)
    assert np.all(X >= spot_instance.lower)
    assert np.all(X <= spot_instance.upper)

def test_generate_design_repeat_effect(spot_instance):
    n_points = 3
    X1 = spot_instance.generate_design(size=n_points, repeats=1, lower=spot_instance.lower, upper=spot_instance.upper)
    X2 = spot_instance.generate_design(size=n_points, repeats=2, lower=spot_instance.lower, upper=spot_instance.upper)
    assert X2.shape[0] == 2 * X1.shape[0]
    assert X2.shape[1] == X1.shape[1]

def test_generate_design_zero_points(spot_instance):
    with pytest.raises(ValueError):
        spot_instance.generate_design(size=0, repeats=1, lower=spot_instance.lower, upper=spot_instance.upper)

def test_generate_design_invalid_bounds(spot_instance):
    lower = np.array([2, 1])
    upper = np.array([-2, -1])
    with pytest.raises(ValueError):
        spot_instance.generate_design(size=3, repeats=1, lower=lower, upper=upper)