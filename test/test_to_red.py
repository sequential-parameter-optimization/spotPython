import numpy as np
from spotpython.fun.objectivefunctions import analytical
from spotpython.spot import spot
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
    fun = analytical().fun_sphere
    lower = np.array([-1, -1, -1])
    upper = np.array([-1, 1, 1])
    spot_1 = spot.Spot(
        fun=fun,
        fun_control=fun_control_init(lower=lower, upper=upper, fun_evals=fun_evals, show_progress=True, log_level=50),
        design_control=design_control_init(init_size=ni),
        surrogate_control=surrogate_control_init(n_theta=2),
    )
    spot_1.run()
    assert spot_1.lower.size == 2
    assert spot_1.upper.size == 2
    assert len(spot_1.var_type) == 2
    assert spot_1.red_dim
