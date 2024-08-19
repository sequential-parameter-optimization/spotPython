def test_show_progress():
    """
    Test show_progress
    """
    import numpy as np
    from spotpython.fun.objectivefunctions import analytical
    from spotpython.spot import spot
    from math import inf
    from spotpython.utils.init import (
        fun_control_init,
        design_control_init,
    )

    # number of initial points:
    ni = 7
    # number of points
    n = 10

    fun = analytical().fun_sphere
    lower = np.array([-1])
    upper = np.array([1])

    spot_1 = spot.Spot(
        fun=fun,
        fun_control=fun_control_init(lower=lower, upper=upper, fun_evals=n, show_progress=False),
        design_control=design_control_init(init_size=ni),
    )
    spot_1.run()
    # To check whether the run was successfully completed,
    # we compare the number of evaluated points to the specicified
    # number of points.
    assert spot_1.y.shape[0] == n

    spot_2 = spot.Spot(
        fun=fun,
        fun_control=fun_control_init(lower=lower, upper=upper, fun_evals=n, show_progress=False),
        design_control=design_control_init(init_size=ni),
    )
    spot_2.run()
    # To check whether the run was successfully completed,
    # we compare the number of evaluated points to the specicified
    # number of points.
    assert spot_2.y.shape[0] == n

    spot_3 = spot.Spot(
        fun=fun,
        fun_control=fun_control_init(lower=lower, upper=upper, fun_evals=inf, max_time=0.1, show_progress=False),
        design_control=design_control_init(init_size=ni),
    )
    spot_3.run()
    # To check whether the run was successfully completed,
    # we test whether the number of evaluated points is greater than zero,
    # because we do not know how many points can be evaluated.
    assert spot_3.y.shape[0] > 0

    spot_4 = spot.Spot(
        fun=fun,
        fun_control=fun_control_init(lower=lower, upper=upper, fun_evals=inf, max_time=0.1, show_progress=True),
        design_control=design_control_init(init_size=ni),
    )
    spot_4.run()
    # To check whether the run was successfully completed,
    # we test whether the number of evaluated points is greater than zero,
    # because we do not know how many points can be evaluated.
    assert spot_4.y.shape[0] > 0
