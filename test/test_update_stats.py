import numpy as np
from spotPython.fun.objectivefunctions import analytical
from spotPython.spot import spot

def update_stats_no_duplicates():
    # number of initial points:
    ni = 0
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    fun = analytical().fun_sphere
    lower = np.array([-1, -1])
    upper = np.array([1, 1])
    design_control={"init_size": ni}
    S = spot.Spot(fun=fun,
                lower = lower,
                upper= upper,
                show_progress=True,
                design_control=design_control,)
    S.initialize_design(X_start=X_start)
    S.update_stats()
    assert np.equal(S.min_X, X_start[0]).all()
    assert S.min_y == fun(X_start[0])
    assert S.counter == X_start.shape[0]
    # Since noise is False, the following statics should be None:
    assert S.mean_X is None
    assert S.mean_y is None
    assert S.var_y is None
    assert S.min_mean_X is None
    assert S.min_mean_y is None

def test_update_stats_duplicates_and_noise():
    # number of initial points:
    ni = 0
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])

    fun = analytical().fun_sphere
    lower = np.array([-1, -1])
    upper = np.array([1, 1])
    design_control={"init_size": ni}

    S = spot.Spot(fun=fun,
                noise=True,
                lower = lower,
                upper= upper,
                show_progress=True,
                design_control=design_control,)
    S.initialize_design(X_start=X_start)
    print(f"S.X: {S.X}")
    print(f"S.y: {S.y}")
    S.update_stats()
    assert np.equal(S.min_X, X_start[0]).all()
    assert S.min_y == fun(X_start[0])
    assert S.counter == X_start.shape[0]
    # the X values are aggregated, the last two rows are equal,
    # so the mean_X should have only 4 rows
    assert np.equal(S.mean_X,
                    np.array([[0., 0.],
                            [0., 1.],
                            [1., 0.],
                            [1., 1.]])).all()
    # the y values are also aggregated, there are only 4 values
    assert np.equal(S.mean_y, np.array([0., 1., 1., 2.])).all()

def test_update_stats_duplicates_nonoise():
    # number of initial points:
    ni = 0
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])

    fun = analytical().fun_sphere
    lower = np.array([-1, -1])
    upper = np.array([1, 1])
    design_control={"init_size": ni}

    S = spot.Spot(fun=fun,
                noise=False,
                lower = lower,
                upper= upper,
                show_progress=True,
                design_control=design_control,)
    S.initialize_design(X_start=X_start)
    print(f"S.X: {S.X}")
    print(f"S.y: {S.y}")
    S.update_stats()
    assert np.equal(S.min_X, X_start[0]).all()
    assert S.min_y == fun(X_start[0])
    assert S.counter == X_start.shape[0]
    # Since noise is False, the following statics should be None:
    assert S.mean_X is None
    assert S.mean_y is None
    assert S.var_y is None
    assert S.min_mean_X is None
    assert S.min_mean_y is None
    # the X values are not aggregated, the last two equal rows ae not modified:
    assert np.equal(S.X,
                    np.array([[0., 0.],
                            [0., 1.],
                            [1., 0.],
                            [1., 1.],
                            [1., 1.]])).all()