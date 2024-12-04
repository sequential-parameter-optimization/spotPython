def test_infill():
    """
    Test infill method
    """
    import numpy as np
    from spotpython.fun.objectivefunctions import Analytical
    from spotpython.spot.spot import Spot
    from spotpython.utils.repair import repair_non_numeric
    from spotpython.utils.init import fun_control_init
    from sklearn import linear_model
    import pytest

    fun = Analytical().fun_sphere

    fun_control = fun_control_init(lower=np.array([-10, -1]), upper=np.array([10, 1]))

    spot_test = Spot(fun=fun, fun_control=fun_control)

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

    x = spot_test.X[0, :]
    y = spot_test.y[0]

    y0 = spot_test.infill(x)
    assert y0.shape[0] == 1
    assert y0.ndim == 1
    # Kriging is interpolating, so y and yhat should be similar:
    assert y0 == pytest.approx(y, 0.1)

    # 2nd test with sklearn surrogate (a simple linear model):

    S_LM = linear_model.LinearRegression()
    spot_test = Spot(fun=fun, fun_control=fun_control, surrogate=S_LM)
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

    x = spot_test.X[0, :]
    y0 = spot_test.infill(x)
    assert y0.shape[0] == 1
    assert y0.ndim == 1
