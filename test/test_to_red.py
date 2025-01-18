import numpy as np
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.utils.init import fun_control_init, surrogate_control_init, design_control_init


def test_to_red():
    """
    Test to_red.
    Test reduced dimensionality.
    The first variable is not active, because it has
    identical values (bounds).

    """
    # number of initial points:
    ni = 10
    # number of points
    fun_evals = 10
    fun = Analytical().fun_sphere
    lower = np.array([-1, -1, -1])
    upper = np.array([-1, 1, 1])
    PREFIX = "test_to_red"
    spot_1 = Spot(
        fun=fun,
        fun_control=fun_control_init(PREFIX=PREFIX,
                                     lower=lower,
                                     upper=upper,
                                     fun_evals=fun_evals,
                                     show_progress=True,
                                     log_level=50),
        design_control=design_control_init(init_size=ni),
        surrogate_control=surrogate_control_init(n_theta=2),
    )
    spot_1.run()
    assert spot_1.lower.size == 2
    assert spot_1.upper.size == 2
    assert len(spot_1.var_type) == 2
    assert spot_1.red_dim


def test_to_red_dim():
    # Setup: Define lower and upper bounds where some dimensions are fixed
    lower = np.array([-1, -1, 0, 0])
    upper = np.array([1, -1, 0, 5])  # Second and third dimensions are fixed
    fun_evals = 10

    var_type = ['float', 'int', 'float', 'int']
    var_name = ['x1', 'x2', 'x3', 'x4']

    spot_instance = Spot(
        # Assuming Spot takes fun, fun_control, design_control as arguments
        fun = Analytical().fun_sphere,  # Replace with appropriate function
        fun_control=fun_control_init(PREFIX = "test_to_red_dim",
                                     lower=lower,
                                     upper=upper,
                                     fun_evals=fun_evals
                                     ),        
    )
    
    spot_instance.var_type = var_type
    spot_instance.var_name = var_name

    # Action: Call the to_red_dim function
    spot_instance.to_red_dim()

    # Assert: Check if dimensions were reduced correctly
    assert spot_instance.lower.size == 2, "Expected size of lower to be 2 after reduction"
    assert spot_instance.upper.size == 2, "Expected size of upper to be 2 after reduction"
    assert len(spot_instance.var_type) == 2, "Expected size of var_type to be 2 after reduction"
    assert spot_instance.k == 2, "Expected k to reflect the reduced dimensions"
    
    # Check remaining values
    expected_lower = np.array([-1, 0])
    expected_upper = np.array([1, 5])
    expected_var_type = ['float', 'int']
    # there are two remaining variables, they are named x1 and x2
    expected_var_name = ['x1', 'x2']
    
    np.testing.assert_array_equal(spot_instance.lower, expected_lower)
    np.testing.assert_array_equal(spot_instance.upper, expected_upper)
    assert spot_instance.var_type == expected_var_type
    assert spot_instance.var_name == expected_var_name
