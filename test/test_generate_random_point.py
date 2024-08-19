import pytest
import numpy as np
from unittest.mock import Mock
from spotpython.spot import spot
from spotpython.utils.init import fun_control_init


@pytest.fixture
def setup_spot():
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )
    return fun_control


def test_generate_random_point(setup_spot):
    fun = Mock(return_value=np.array([0]))  # Replace with valid function
    S = spot.Spot(fun=fun, fun_control=setup_spot)
    X0, y0 = S.generate_random_point()

    print(f"X0: {X0}")
    print(f"y0: {y0}")

    assert X0.size == 2, "X0 does not have a size of 2"
    assert y0.size == 1, "y0 does not have a size of 1"
    assert np.all(X0 >= S.lower), "X0 has values less than lower bounds"
    assert np.all(X0 <= S.upper), "X0 has values greater than upper bounds"
    assert y0 >= 0, "y0 is not greater than or equal to 0"


def test_generate_random_point_with_nan(setup_spot):
    fun = Mock(return_value=np.array([np.nan]))  # Function that returns NaN
    S = spot.Spot(fun=fun, fun_control=setup_spot)
    X0, y0 = S.generate_random_point()

    print(f"X0 with NaN: {X0}")
    print(f"y0 with NaN: {y0}")

    assert X0.shape[0] == 0, "X0 should have shape[0] == 0 when fun returns NaN"
    assert y0.shape[0] == 0, "y0 should have shape[0] == 0 when fun returns NaN"
