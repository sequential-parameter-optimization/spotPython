import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, optimizer_control_init, surrogate_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def test_surrogate_control_setup_sets_var_type():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]), var_type=["num", "int"])
    surrogate_control = surrogate_control_init(var_type=None)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, surrogate_control=surrogate_control)
    # _surrogate_control_setup is called in __init__, but let's call it again to check idempotency
    spot._surrogate_control_setup()
    assert spot.surrogate_control["var_type"] == ["num", "int"]

def test_surrogate_control_setup_sets_method_from_fun_control():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    fun_control["method"] = "interpolation"  # <-- FIXED: use key access
    surrogate_control = surrogate_control_init(method=None)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, surrogate_control=surrogate_control)
    spot._surrogate_control_setup()
    assert spot.surrogate_control["method"] == "interpolation"

def test_surrogate_control_setup_sets_model_fun_evals_from_optimizer_control():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    optimizer_control = optimizer_control_init(max_iter=1234)
    surrogate_control = surrogate_control_init(model_fun_evals=None)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, optimizer_control=optimizer_control, surrogate_control=surrogate_control)
    spot._surrogate_control_setup()
    assert spot.surrogate_control["model_fun_evals"] == 1234

def test_surrogate_control_setup_sets_model_optimizer():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    surrogate_control = surrogate_control_init(model_optimizer=None)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, surrogate_control=surrogate_control)
    spot._surrogate_control_setup()
    # Should be set to spot.optimizer (which is not None after __init__)
    assert spot.surrogate_control["model_optimizer"] == spot.optimizer

def test_surrogate_control_setup_updates_model_optimizer_if_optimizer_is_not_none():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    surrogate_control = surrogate_control_init(model_optimizer="old_optimizer")
    spot = Spot(fun=dummy_fun, fun_control=fun_control, surrogate_control=surrogate_control)
    # Set a new optimizer
    def custom_optimizer(*args, **kwargs):
        return "custom"
    spot.optimizer = custom_optimizer
    spot._surrogate_control_setup()
    assert spot.surrogate_control["model_optimizer"] == custom_optimizer