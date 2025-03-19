import numpy as np
from spotpython.spot.spot import Spot

class MockSpot(Spot):
    def __init__(self, fun_control):
        self.fun_control = fun_control

def test_mo2so():
    # Case 1: Multi-objective with a user-defined function
    fun_control = {"fun_mo2so": lambda y: np.sum(y, axis=0)}  # Sum across objectives
    spot_instance = MockSpot(fun_control=fun_control)
    y_mo = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
    result = spot_instance._mo2so(y_mo)
    expected = np.array([5, 7, 9])  # Sum of rows
    np.testing.assert_array_equal(result, expected)

    # Case 2: Multi-objective without a user-defined function
    fun_control = {"fun_mo2so": None}  # No function provided
    spot_instance = MockSpot(fun_control=fun_control)
    y_mo = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
    result = spot_instance._mo2so(y_mo)
    expected = np.array([1, 2, 3])  # First row
    np.testing.assert_array_equal(result, expected)

    # Case 3: Single-objective (no transformation needed)
    y_mo = np.array([[1, 2, 3]])  # Shape (1, 3)
    result = spot_instance._mo2so(y_mo)
    expected = np.array([[1, 2, 3]])  # No change
    np.testing.assert_array_equal(result, expected)

    # Case 4: Single-objective with a single data point
    y_mo = np.array([[1]])  # Shape (1, 1)
    result = spot_instance._mo2so(y_mo)
    expected = np.array([[1]])  # No change
    np.testing.assert_array_equal(result, expected)

    # Case 5: Empty input
    y_mo = np.array([[]])  # Shape (1, 0)
    result = spot_instance._mo2so(y_mo)
    expected = np.array([[]])  # No change
    np.testing.assert_array_equal(result, expected)