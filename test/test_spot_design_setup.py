import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def test_design_setup_default():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    # _design_setup is called in __init__, but let's call it again to check idempotency
    spot._design_setup(None)
    # Should use SpaceFilling by default
    from spotpython.design.spacefilling import SpaceFilling
    assert isinstance(spot.design, SpaceFilling)
    assert spot.design.k == 2

def test_design_setup_custom_object():
    fun_control = fun_control_init(lower=np.array([0, 0]), upper=np.array([1, 1]))
    class DummyDesign:
        def __init__(self, k):
            self.k = k
    custom_design = DummyDesign(k=2)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._design_setup(custom_design)
    assert spot.design is custom_design
    assert spot.design.k == 2

def test_design_setup_with_different_k():
    fun_control = fun_control_init(lower=np.array([0, 0, 0]), upper=np.array([1, 1, 1]))
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._design_setup(None)
    assert spot.design.k == 3