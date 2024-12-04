def test_evaluate_new_X():
    """
    Test evaluation of suggest_new_X points suggested by the surrogate
    """
    import numpy as np
    from spotpython.spot import spot
    from spotpython.fun.objectivefunctions import Analytical
    from spotpython.utils.init import (
        fun_control_init,
    )

    nn = 3
    fun_sphere = Analytical().fun_sphere
    fun_control = fun_control_init(lower=np.array([-1, -1]), upper=np.array([1, 1]), n_points=nn)
    spot_1 = spot.Spot(
        fun=fun_sphere,
        fun_control=fun_control,
    )

    # (S-2) Initial Design:
    spot_1.X = spot_1.design.scipy_lhd(spot_1.design_control["init_size"], lower=spot_1.lower, upper=spot_1.upper)
    print(spot_1.X)

    # (S-3): Eval initial design:
    spot_1.y = spot_1.fun(spot_1.X)
    print(spot_1.y)

    spot_1.surrogate.fit(spot_1.X, spot_1.y)
    X0 = spot_1.suggest_new_X()
    X0.size
    assert X0.size == spot_1.n_points * spot_1.k
    assert X0.ndim == 2
    assert X0.shape[0] == nn
    assert X0.shape[1] == 2
