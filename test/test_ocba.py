def test_ocba():
    """
    Test OCBA.

    """

import copy
import numpy as np
from math import inf
from spotPython.fun.objectivefunctions import analytical
from spotPython.spot import spot
from scipy.optimize import shgo
from scipy.optimize import direct
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from numpy import float64
from spotPython.budget.ocba import get_ocba

# Test based on the example from the book:
# Chun-Hung Chen and Loo Hay Lee:
#     Stochastic Simulation Optimization: An Optimal Computer Budget Allocation,
#     pp. 49 and pp. 215
#     p. 49:
#     mean_y = np.array([1,2,3,4,5])
#     var_y = np.array([1,1,9,9,4])
#     get_ocba(mean_y, var_y, 50)
#     [11  9 19  9  2]

fun = analytical().fun_linear
fun_control = {"sigma": 0.001,
               "seed": 123}
spot_1_noisy = spot.Spot(fun=fun,
                   lower = np.array([-1]),
                   upper = np.array([1]),
                   fun_evals = 20,
                   fun_repeats = 2,
                   noise = True,
                   ocba_delta=1,
                   seed=123,
                   show_models=False,
                   fun_control = fun_control,
                   design_control={"init_size": 3,
                                   "repeats": 2},
                   surrogate_control={"noise": True})
spot_1_noisy.run()
spot_2 = copy.deepcopy(spot_1_noisy)
spot_2.mean_y = np.array([1,2,3,4,5])
spot_2.var_y = np.array([1,1,9,9,4])
n = 50
o = get_ocba(spot_2.mean_y, spot_2.var_y, n)
assert sum(o) == 50
assert (o == np.array([[11, 9, 19, 9, 2]])).all()