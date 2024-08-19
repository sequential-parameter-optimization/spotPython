def test_repair_non_numeric():
    """
    Test repair_non_numeric method
    """
    import numpy as np
    from spotpython.fun.objectivefunctions import analytical
    from spotpython.spot.spot import Spot
    from spotpython.utils.repair import repair_non_numeric
    from spotpython.utils.init import (
        fun_control_init,
        design_control_init,
    )

    fun = analytical().fun_branin_factor
    ni = 12
    spot_test = Spot(
        fun=fun,
        fun_control=fun_control_init(
            lower=np.array([-5, -0, 1]), upper=np.array([10, 15, 3]), var_type=["num", "num", "factor"]
        ),
        design_control=design_control_init(init_size=ni),
    )
    spot_test.run()
    # 3rd variable should be a rounded float, because it was labeled as a factor
    assert spot_test.min_X[2] == round(spot_test.min_X[2])

    spot_test.X = spot_test.generate_design(
        size=spot_test.design_control["init_size"],
        repeats=spot_test.design_control["repeats"],
        lower=spot_test.lower,
        upper=spot_test.upper,
    )
    spot_test.X = repair_non_numeric(spot_test.X, spot_test.var_type)
    assert spot_test.X.ndim == 2
    assert spot_test.X.shape[0] == ni
    assert spot_test.X.shape[1] == 3
