import numpy as np
from spotpython.fun.objectivefunctions import analytical
from spotpython.spot import spot
from spotpython.utils.init import fun_control_init, design_control_init


def test_update_design():
    # number of initial points:
    ni = 0
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
    fun = analytical().fun_sphere
    lower = np.array([-1, -1])
    upper = np.array([1, 1])
    S = spot.Spot(
        fun=fun,
        fun_control=fun_control_init(noise=False, lower=lower, upper=upper, show_progress=True),
        design_control=design_control_init(init_size=ni),
    )
    S.initialize_design(X_start=X_start)
    X_shape_before = S.X.shape
    y_size_before = S.y.size
    S.update_stats()
    S.fit_surrogate()
    S.update_design()
    # compare the shapes of the X and y values before and after the update_design method
    assert X_shape_before[0] + S.n_points == S.X.shape[0]
    assert X_shape_before[1] == S.X.shape[1]
    assert y_size_before + S.n_points == S.y.size


def test_update_design_with_repeats_and_ocba():
    # number of initial points:
    ni = 3
    X_start = np.array([[1, 0], [1, 0], [1, 1], [1, 1]])
    fun = analytical().fun_sphere
    fun_control = fun_control_init(
        sigma=0.02,
        seed=123,
        noise=True,
        fun_repeats=2,
        n_points=1,
        ocba_delta=1,
        show_progress=True,
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
    )
    S = spot.Spot(fun=fun, fun_control=fun_control, design_control=design_control_init(init_size=ni, repeats=2))
    S.initialize_design(X_start=X_start)
    X_shape_before = S.X.shape
    y_size_before = S.y.size
    S.update_stats()
    S.fit_surrogate()
    S.update_design()
    # compare the shapes of the X and y values before and after the update_design method
    assert X_shape_before[0] + S.n_points * S.fun_repeats + S.ocba_delta == S.X.shape[0]
    assert X_shape_before[1] == S.X.shape[1]
    assert y_size_before + S.n_points * S.fun_repeats + S.ocba_delta == S.y.size


def test_update_design_with_repeats_and_ocba_no_var():
    # number of initial points:
    ni = 3
    # The first two X_start points have no repeats and therefore no variance
    X_start = np.array([[0, 1], [1, 0], [1, 1], [1, 1]])
    fun = analytical().fun_sphere
    fun_control = fun_control_init(
        sigma=0.02,
        seed=123,
        fun_repeats=2,
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        noise=True,
        n_points=1,
        ocba_delta=1,
        show_progress=True,
    )
    S = spot.Spot(fun=fun, fun_control=fun_control, design_control=design_control_init(init_size=ni, repeats=2))
    S.initialize_design(X_start=X_start)
    X_shape_before = S.X.shape
    y_size_before = S.y.size
    S.update_stats()
    S.fit_surrogate()
    S.update_design()
    # compare the shapes of the X and y values before and after the update_design method
    # no ocba points are added because the first two points have no variance
    assert X_shape_before[0] + S.n_points * S.fun_repeats == S.X.shape[0]
    assert X_shape_before[1] == S.X.shape[1]
    assert y_size_before + S.n_points * S.fun_repeats == S.y.size
