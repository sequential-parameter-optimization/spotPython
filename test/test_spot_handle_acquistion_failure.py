import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def dummy_fun(X, fun_control=None):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1)

def test_handle_acquisition_failure_random_strategy(monkeypatch):
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    var_type = ['float', 'float']
    var_name = ['x1', 'x2']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    spot.acquisition_failure_strategy = "random"
    spot.X = np.array([[0.1, 0.2], [0.3, 0.4]])
    # Should generate a new design in the correct shape and bounds
    X0 = spot._handle_acquisition_failure()
    assert isinstance(X0, np.ndarray)
    assert X0.shape[1] == spot.k
    assert np.all(X0 >= spot.lower)
    assert np.all(X0 <= spot.upper)

def test_handle_acquisition_failure_mm_strategy(monkeypatch):
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    var_type = ['float', 'float']
    var_name = ['x1', 'x2']
    fun_control = fun_control_init(lower=lower, upper=upper, var_type=var_type, var_name=var_name)
    design_control = design_control_init(init_size=3, repeats=2)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    spot.acquisition_failure_strategy = "mm"
    spot.X = np.array([[0.1, 0.2], [0.3, 0.4]])
    # Should generate a new mmphi point, repeated as needed
    X0 = spot._handle_acquisition_failure()
    assert isinstance(X0, np.ndarray)
    assert X0.shape[1] == spot.k
    # Should be repeated according to repeats
    assert X0.shape[0] % spot.design_control["repeats"] == 0
    assert np.all(X0 >= spot.lower)
    assert np.all(X0 <= spot.upper)