import numpy as np
from spotPython.fun.objectivefunctions import analytical
from spotPython.spot import spot

def test_fit_surrogate():
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
    S.update_stats()
    S.fit_surrogate()
    # correlation matrix should be square and the same size as the number of points
    assert S.surrogate.Psi.shape[0] == S.X.shape[0]