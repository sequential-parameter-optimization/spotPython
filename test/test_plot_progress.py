import numpy as np
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.utils.init import fun_control_init, surrogate_control_init, design_control_init


def test_plot_progress():
    # number of initial points:
    ni = 7
    # number of points
    fun_evals = 10
    fun = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1]), upper=np.array([1, 1]), fun_evals=fun_evals, tolerance_x=np.sqrt(np.spacing(1))
    )
    design_control = design_control_init(init_size=ni)
    surrogate_control = surrogate_control_init(n_theta=3)
    S = Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control,
    )
    S.run()

    # Test plot_progress with different parameters
    S.plot_progress(show=False)  # Test with show=False
    S.plot_progress(log_x=True, show=False)  # Test with log_x=True
    S.plot_progress(log_y=True, show=False)  # Test with log_y=True
    S.plot_progress(filename="test_plot.png", show=False)  # Test with a different filename
    # add NaN to S.y at position 2
    S.y[2] = np.nan
    S.plot_progress(show=False)  # Test with NaN in S.y


def test_plot_progress_n_init():
    # number of initial points:
    ni = 7
    # number of points
    fun_evals = 10
    fun = Analytical().fun_sphere
    fun_control = fun_control_init(
        lower=np.array([-1, -1]), upper=np.array([1, 1]), fun_evals=fun_evals, tolerance_x=np.sqrt(np.spacing(1))
    )
    design_control = design_control_init(init_size=ni)
    surrogate_control = surrogate_control_init(n_theta=3)
    S = Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
        surrogate_control=surrogate_control,
    )
    S.run()

    # remove points from S.y so that there are less than ni points
    S.y = S.y[:3]
    # Test plot_progress with different parameters
    S.plot_progress(show=False)  # Test with show=False
    S.plot_progress(log_x=True, show=False)  # Test with log_x=True
    S.plot_progress(log_y=True, show=False)  # Test with log_y=True
    S.plot_progress(filename="test_plot.png", show=False)  # Test with a different filename
