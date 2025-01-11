import pytest
import numpy as np
from unittest.mock import MagicMock
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, design_control_init
from spotpython.spot import spot  # Import spot as module, assuming it contains Spot

def create_spot_instance():
    # Number of initial points
    ni = 3
    X_start = np.array([[0, 1], [1, 0], [1, 1], [1, 1]])
    fun = Analytical().fun_sphere
    fun_control = fun_control_init(
        n_points=10,
        ocba_delta=0,
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )
    design_control = design_control_init(init_size=ni)
    spot_instance = spot.Spot(fun=fun, fun_control=fun_control, design_control=design_control)
    spot_instance.initialize_design(X_start=X_start)
    return spot_instance

def test_get_new_X0_successful_suggestion():
    spot_instance = create_spot_instance()

    # Configure essential testing properties
    spot_instance.tolerance_x = 0.1
    spot_instance.var_type = ['float', 'float']
    spot_instance.X = np.array([[0, 0], [1, 1]])

    # Mock the methods to simulate expected behaviors
    spot_instance.suggest_new_X = MagicMock(return_value=np.array([[0.5, 0.5]] * 10))  # Must return 'n_points' amount
    spot_instance.repair_non_numeric = MagicMock(side_effect=lambda X, var_type: X)
    spot_instance.selectNew = MagicMock(return_value=(np.array([[0.5, 0.5]] * 10), np.arange(10)))

    # Call the method under test
    X0 = spot_instance.get_new_X0()

    # Assertions
    assert X0.shape[0] == spot_instance.fun_control['n_points']
    assert np.all(X0 >= spot_instance.fun_control['lower']) and np.all(X0 <= spot_instance.fun_control['upper'])

def test_get_new_X0_with_space_filling():
    spot_instance = create_spot_instance()

    # Configure essential testing properties
    spot_instance.tolerance_x = 0.1
    spot_instance.var_type = ['float', 'float']
    spot_instance.X = np.array([[0, 0], [1, 1]])

    # Mock methods with appropriate return values
    spot_instance.suggest_new_X = MagicMock(return_value=np.empty((0, 2)))  # Ensure it's a 2D array with 0 rows
    spot_instance.repair_non_numeric = MagicMock(side_effect=lambda X, var_type: X)
    spot_instance.selectNew = MagicMock(return_value=(np.empty((0, 2)), []))
    spot_instance.generate_design = MagicMock(return_value=np.array([[-0.5, 0.5]] * 10))  # Must return 'n_points'

    # Call the method under test
    X0 = spot_instance.get_new_X0()

    # Assertions
    assert X0.shape[0] == spot_instance.fun_control['n_points']
    assert np.all(X0 >= spot_instance.fun_control['lower']) and np.all(X0 <= spot_instance.fun_control['upper'])