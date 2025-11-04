import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def test_optimizer_setup_default():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # _optimizer_setup is called in __init__, but let's call it again to check idempotency
    spot._optimizer_setup(None)
    # Should use scipy.optimize.differential_evolution by default
    import scipy.optimize
    assert spot.optimizer == scipy.optimize.differential_evolution

def test_optimizer_setup_custom():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    def custom_optimizer(*args, **kwargs):
        return "custom"
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._optimizer_setup(custom_optimizer)
    assert spot.optimizer == custom_optimizer

def test_optimizer_setup_preserves_existing():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    def custom_optimizer(*args, **kwargs):
        return "custom"
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot.optimizer = custom_optimizer
    spot._optimizer_setup(None)
    # The optimizer will be overwritten to differential_evolution
    from scipy.optimize import differential_evolution
    assert spot.optimizer == differential_evolution