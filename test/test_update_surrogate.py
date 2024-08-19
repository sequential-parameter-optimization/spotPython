def test_update_statss():
    """
    Test update surrogate method
    """
    import numpy as np
    from spotpython.fun.objectivefunctions import analytical
    from spotpython.spot.spot import Spot
    from spotpython.utils.repair import repair_non_numeric
    from spotpython.utils.init import (
        fun_control_init,
        design_control_init,
    )

    fun = analytical().fun_sphere

    nn = 3
    ni = 5

    spot_test = Spot(
        fun=fun,
        fun_control=fun_control_init(lower=np.array([-10, -1]), upper=np.array([10, 1]), n_points=nn),
        design_control=design_control_init(init_size=ni),
    )

    # (S-2) Initial Design:
    spot_test.X = spot_test.generate_design(
        size=spot_test.design_control["init_size"],
        repeats=spot_test.design_control["repeats"],
        lower=spot_test.lower,
        upper=spot_test.upper,
    )
    spot_test.X = repair_non_numeric(spot_test.X, spot_test.var_type)

    # (S-3): Eval initial design:
    spot_test.y = spot_test.fun(spot_test.X)
    spot_test.surrogate.fit(spot_test.X, spot_test.y)
    X0 = spot_test.suggest_new_X()
    X0 = repair_non_numeric(X0, spot_test.var_type)

    # (S-18): Evaluating New Solutions:
    y0 = spot_test.fun(X0)
    spot_test.X = np.append(spot_test.X, X0, axis=0)
    spot_test.y = np.append(spot_test.y, y0)
    spot_test.update_stats()
    # (S-11) Surrogate Fit:
    spot_test.surrogate.fit(spot_test.X, spot_test.y)

    # Check whether X0 is appended to spot_test.X:
    assert (spot_test.X[spot_test.counter - nn : spot_test.counter, :] == X0).any()

    # Check dimensions
    assert spot_test.X.shape[0] == nn + ni
    assert spot_test.X.shape[1] == 2
    assert spot_test.y.ndim == 1
    assert spot_test.y.shape[0] == nn + ni
