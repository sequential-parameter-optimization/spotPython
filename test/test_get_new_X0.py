import numpy as np
from spotPython.fun.objectivefunctions import analytical
from spotPython.spot import spot
from spotPython.utils.init import fun_control_init

def test_get_new_X0():
    # number of initial points:
    ni = 3
    X_start = np.array([[0, 1], [1, 0], [1, 1], [1, 1]])

    fun = analytical().fun_sphere
    fun_control = fun_control_init(
            sigma=0.0,
            seed=123,)
    lower = np.array([-1, -1])
    upper = np.array([1, 1])
    design_control={"init_size": ni,
                    "repeats": 1}

    S = spot.Spot(fun=fun,
                noise=False,
                fun_repeats=1,
                n_points=10,
                ocba_delta=0,
                lower = lower,
                upper= upper,
                show_progress=True,
                design_control=design_control,
                fun_control=fun_control
    )
    S.initialize_design(X_start=X_start)
    S.update_stats()
    S.fit_surrogate()
    X0 = S.get_new_X0()
    assert X0.shape[0] == S.n_points
    assert X0.shape[1] == S.lower.size
    # assert new points are in the interval [lower, upper]
    assert np.all(X0 >= S.lower)
    assert np.all(X0 <= S.upper)
