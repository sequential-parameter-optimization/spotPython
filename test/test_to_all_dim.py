import pytest
import numpy as np
from spotpython.fun import Analytical
from spotpython.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init, optimizer_control_init, surrogate_control_init

def test_to_all_dim():
    # Initialize the Spot instance with necessary parameters
    lower = np.array([-1, -1, 0, 0])
    upper = np.array([1, -1, 0, 5])  # Second and third dimensions are fixed
    fun_control = fun_control_init(lower=lower, upper=upper)
    design_control = design_control_init()
    optimizer_control = optimizer_control_init()
    surrogate_control = surrogate_control_init()
    
    spot_instance = Spot(
        fun = Analytical().fun_sphere,  # Dummy function
        fun_control=fun_control,
        design_control=design_control,
        optimizer_control=optimizer_control,
        surrogate_control=surrogate_control
    )
    
    # Manually set the attributes required for the test
    spot_instance.ident = np.array([False, True, True, False])
    spot_instance.all_lower = lower
    
    # Reduced dimension design points
    X0 = np.array([[0.5, 4.0], [-0.5, 1.0]])
    
    # Expected full dimension design points
    expected_X = np.array([[0.5, -1, 0, 4.0], [-0.5, -1, 0, 1.0]])
    
    # Call the method
    result_X = spot_instance.to_all_dim(X0)
    
    # Assert the result
    np.testing.assert_array_equal(result_X, expected_X)

def test_to_all_dim_2():
    # Setup for the Spot instance and necessary attributes
    lower = np.array([-1, -1, 0, 0])
    upper = np.array([1, -1, 0, 5])
    ident = np.array([False, True, True, False])  # Fixed dimensions

    # Create a Spot instance and set the relevant properties
    spot_instance = Spot(fun = Analytical().fun_sphere,
                        fun_control=fun_control_init(lower=lower, upper=upper))
    spot_instance.ident = ident
    spot_instance.all_lower = np.array([-1, -1, 0, 0])

    # Define reduced dimension design points X0
    X0 = np.array([[2.5, 3.5], [4.5, 5.5]])

    # Call the to_all_dim method
    X_full_dim = spot_instance.to_all_dim(X0)

    # Expected result after reinserting fixed dimensions
    expected_output = np.array([[2.5, -1, 0, 3.5],
                                [4.5, -1, 0, 5.5]])

    # Assert to verify the output matches the expected result
    np.testing.assert_array_equal(X_full_dim, expected_output)    
