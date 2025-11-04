import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def test_update_stats_basic():
    # Simple quadratic function, single objective
    def fun(X, **kwargs):
        return np.sum(X**2, axis=1)
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        fun_evals=5,
        noise=False
    )
    design_control = design_control_init(init_size=5)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    spot.update_stats()
    # Check min_y and min_X
    assert np.isclose(spot.min_y, np.min(spot.y))
    np.testing.assert_array_equal(spot.min_X, spot.X[np.argmin(spot.y)])
    # Check counter
    assert spot.counter == spot.y.size

def test_update_stats_with_noise():
    # Function with noise, so mean and var stats are computed
    def fun(X, **kwargs):
        rng = np.random.default_rng(42)
        return np.sum(X**2, axis=1) + rng.normal(0, 0.1, size=X.shape[0])
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        fun_evals=6,
        noise=True
    )
    design_control = design_control_init(init_size=3, repeats=2)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    spot.update_stats()
    # Check mean_X, mean_y, var_y shapes
    assert spot.mean_X.shape[0] == spot.mean_y.shape[0] == spot.var_y.shape[0]
    # Check min_mean_y and min_mean_X
    assert np.isclose(spot.min_mean_y, np.min(spot.mean_y))
    np.testing.assert_array_equal(spot.min_mean_X, spot.mean_X[np.argmin(spot.mean_y)])

def test_update_stats_min_y_and_counter():
    # Check that min_y and counter are updated correctly after adding new data
    def fun(X, **kwargs):
        return np.sum(X, axis=1)
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        fun_evals=4,
        noise=False
    )
    design_control = design_control_init(init_size=4)
    spot = Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()
    spot.update_stats()
    # Add a new point with a lower y value
    new_X = np.array([[0, 0]])
    new_y = np.array([0])
    spot.X = np.vstack([spot.X, new_X])
    spot.y = np.append(spot.y, new_y)
    spot.update_stats()
    assert spot.min_y == 0
    np.testing.assert_array_equal(spot.min_X, new_X[0])
    assert spot.counter == spot.y.size