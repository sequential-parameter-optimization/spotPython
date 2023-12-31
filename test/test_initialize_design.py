import numpy as np
from spotPython.fun.objectivefunctions import analytical
from spotPython.spot import spot

def test_initialize_design():
    # number of initial points:
    ni = 7
    fun = analytical().fun_sphere
    lower = np.array([-1])
    upper = np.array([1])
    design_control={"init_size": ni}
    S = spot.Spot(fun=fun,
                lower = lower,
                upper= upper,
                show_progress=False,
                design_control=design_control,)
    S.initialize_design()
    assert S.X.shape[0] == ni
    assert S.X.shape[1] == lower.size

def test_initialize_design_2():
    # number of initial points:
    ni = 7
    # start point
    X_start = np.array([0, 0.5])
    fun = analytical().fun_sphere
    lower = np.array([-1, -1])
    upper = np.array([1, 2])
    design_control={"init_size": ni}

    S = spot.Spot(fun=fun,
                lower = lower,
                upper= upper,
                show_progress=True,
                design_control=design_control,)
    S.initialize_design()
    assert S.X.shape[0] == ni
    assert S.X.shape[1] == lower.size

def test_initialize_design_3():
    # number of initial points:
    ni = 7
    # start point
    X_start = np.array([0, 0]).reshape(1, -1)
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
    assert S.X.shape[0] == ni + np.atleast_2d(X_start).shape[0]
    assert S.X.shape[1] == lower.size

def test_initialize_design_4():
    # number of initial points:
    ni = 7
    # start point
    X_start = np.array([0, 0])
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
    assert S.X.shape[0] == ni + np.atleast_2d(X_start).shape[0]
    assert S.X.shape[1] == lower.size

def test_initialize_design_5():
    # number of initial points:
    ni = 7
    # start point
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
    assert S.X.shape[0] == ni + np.atleast_2d(X_start).shape[0]
    assert S.X.shape[1] == lower.size

def test_initialize_design_noX0():
    # number of initial points is zero, but a start point is given
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
    assert S.X.shape[0] == ni + np.atleast_2d(X_start).shape[0]
    assert S.X.shape[1] == lower.size