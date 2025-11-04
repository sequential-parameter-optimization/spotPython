import numpy as np
from math import inf
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.surrogate.kriging import Kriging
fun = Analytical().fun_branin
from spotpython.utils.init import fun_control_init, design_control_init
# Needed for the sklearn surrogates:
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model


PREFIX = "04"
fun_control = fun_control_init(
    PREFIX=PREFIX,
    lower = np.array([-5,-0]),
    upper = np.array([10,15]),
    fun_evals=10,
    max_time=1)

design_control = design_control_init(
    init_size=10)

S_0 = Kriging(name='kriging', seed=123)


def test_spot_2_GP():
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    S_GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    isinstance(S_GP, GaussianProcessRegressor) 
    isinstance(S_0, Kriging)
    fun = Analytical(seed=123).fun_branin
    spot_2_GP = Spot(fun=fun,
                        fun_control=fun_control,
                        design_control=design_control,
                        surrogate = S_GP)
    spot_2_GP.run()
    success_GP = True
    assert success_GP is True