def test_generate_design():
    """
    Test generate_design method
    """
    import numpy as np
    from spotpython.fun.objectivefunctions import analytical
    from spotpython.spot.spot import Spot

    fun = analytical().fun_branin_factor
    from spotpython.utils.init import (
        fun_control_init,
        design_control_init,
    )

    ni = 7

    spot_test = Spot(
        fun=fun,
        fun_control=fun_control_init(lower=np.array([-5, 0, 1]), upper=np.array([10, 15, 3]), seed=1),
        design_control=design_control_init(init_size=ni),
    )

    X = spot_test.generate_design(
        size=spot_test.design_control["init_size"],
        repeats=spot_test.design_control["repeats"],
        lower=spot_test.lower,
        upper=spot_test.upper,
    )
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            assert X[j, i] >= spot_test.lower[i]
            assert X[j, i] <= spot_test.upper[i]
    assert X.shape[0] == ni
    assert X.shape[1] == 3

    X2 = spot_test.generate_design(
        size=spot_test.design_control["init_size"],
        repeats=spot_test.design_control["repeats"],
        lower=spot_test.lower,
        upper=spot_test.upper,
    )

    spot_test = Spot(
        fun=fun,
        fun_control=fun_control_init(lower=np.array([-5, 0, 1]), upper=np.array([10, 15, 3]), seed=1),
        design_control=design_control_init(init_size=ni),
    )

    X3 = spot_test.generate_design(
        size=spot_test.design_control["init_size"],
        repeats=spot_test.design_control["repeats"],
        lower=spot_test.lower,
        upper=spot_test.upper,
    )

    spot_test = Spot(
        fun=fun,
        fun_control=fun_control_init(lower=np.array([-5, 0, 1]), upper=np.array([10, 15, 3]), seed=2),
        design_control=design_control_init(init_size=ni),
    )

    X4 = spot_test.generate_design(
        size=spot_test.design_control["init_size"],
        repeats=spot_test.design_control["repeats"],
        lower=spot_test.lower,
        upper=spot_test.upper,
    )

    assert (X != X2).any()
    assert (X == X3).any()
    assert (X != X4).any()
    assert (X2 != X3).any()
    assert (X2 != X4).any()
    assert (X3 != X4).any()
