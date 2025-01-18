def test_repair_non_numeric():
    """
    Test repair_non_numeric method
    """
    import numpy as np
    from spotpython.fun.objectivefunctions import Analytical
    from spotpython.spot.spot import Spot
    from spotpython.utils.repair import repair_non_numeric
    from spotpython.utils.init import (
        fun_control_init,
        design_control_init,
    )
    
    fun = Analytical().fun_branin_factor
    ni = 12
    S = Spot(
        fun=fun,
        fun_control=fun_control_init(
            PREFIX="test_repair_non_numeric",
            lower=np.array([-5, -0, 1]),
            upper=np.array([10, 15, 3]),
            var_type=["num", "num", "factor"]
        ),
        design_control=design_control_init(init_size=ni),
    )
    S.run()
    # 3rd variable should be a rounded float, because it was labeled as a factor
    assert S.min_X[2] == round(S.min_X[2])

    S.X = S.generate_design(
        size=S.design_control["init_size"],
        repeats=S.design_control["repeats"],
        lower=S.lower,
        upper=S.upper,
    )
    S.X = repair_non_numeric(S.X, S.var_type)
    assert S.X.ndim == 2
    assert S.X.shape[0] == ni
    assert S.X.shape[1] == 3
