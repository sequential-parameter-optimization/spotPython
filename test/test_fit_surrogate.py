import numpy as np
from spotpython.fun.objectivefunctions import analytical
from spotpython.spot import spot
from spotpython.utils.init import fun_control_init, design_control_init


def test_fit_surrogate():
    # number of initial points:
    ni = 0
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
    fun = analytical().fun_sphere
    fun_control = fun_control_init(
        noise=False,
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        show_progress=True,
    )
    design_control = design_control_init(init_size=ni)
    S = spot.Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
    )
    S.initialize_design(X_start=X_start)
    S.update_stats()
    S.fit_surrogate()
    # correlation matrix should be square and the same size as the number of points
    assert S.surrogate.Psi.shape[0] == S.X.shape[0]
