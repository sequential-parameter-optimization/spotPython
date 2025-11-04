import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def dummy_fun(X, fun_control=None):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1)

def test_copy_from_copies_attributes():
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    fun_control = fun_control_init(lower=lower, upper=upper, fun_evals=3)
    design_control = design_control_init(init_size=2)
    spot1 = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    spot1.X = np.array([[0.1, 0.2], [0.3, 0.4]])
    spot1.y = np.array([0.3, 0.7])
    spot1.min_y = 0.3
    spot1.min_X = np.array([0.1, 0.2])
    spot1.some_custom_attr = "custom_value"

    spot2 = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    # Change spot2's attributes to something else
    spot2.X = np.array([[9.9, 9.9]])
    spot2.y = np.array([99.9])
    spot2.min_y = 99.9
    spot2.min_X = np.array([9.9, 9.9])
    spot2.some_custom_attr = "other_value"

    # Copy from spot1 to spot2
    spot2._copy_from(spot1)

    # Check that all relevant attributes are now equal
    np.testing.assert_array_equal(spot2.X, spot1.X)
    np.testing.assert_array_equal(spot2.y, spot1.y)
    assert spot2.min_y == spot1.min_y
    np.testing.assert_array_equal(spot2.min_X, spot1.min_X)
    assert spot2.some_custom_attr == spot1.some_custom_attr

def test_copy_from_adds_missing_attributes():
    lower = np.array([0])
    upper = np.array([1])
    fun_control = fun_control_init(lower=lower, upper=upper, fun_evals=2)
    spot1 = Spot(fun=dummy_fun, fun_control=fun_control)
    spot1.new_attr = 42

    spot2 = Spot(fun=dummy_fun, fun_control=fun_control)
    assert not hasattr(spot2, "new_attr")
    spot2._copy_from(spot1)
    assert hasattr(spot2, "new_attr")
    assert spot2.new_attr == 42