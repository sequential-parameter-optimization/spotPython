import copy
import numpy as np
from spotPython.fun.objectivefunctions import analytical
from spotPython.spot import spot
from spotPython.budget.ocba import get_ocba, get_ocba_X
from spotPython.utils.aggregate import aggregate_mean_var

def test_ocba():
    """
    Test OCBA.

    """
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

def test_ocba_none_result():
    """
    Test OCBA with None result.

    """
    X = np.array([[1,2,3],
                [1,2,3],
                [4,5,6],
                [4,5,6],
                [4,5,6],
                [4, 5, 6],
                [4, 5 ,6],])
    y = np.array([1,2,30,40, 40, 500, 600  ])
    Z = aggregate_mean_var(X=X, y=y)
    mean_X = Z[0]
    mean_y = Z[1]
    var_y = Z[2]
    print(f"X: {X}")
    print((f"mean_X.shape: {mean_X.shape}"))
    print(f"y: {y}")
    print(f"mean_X: {mean_X}")
    print(f"mean_y: {mean_y}")
    print(f"var_y: {var_y}")
    delta = 5
    # get_ocba(means, vars, delta,verbose=True)
    X_new = get_ocba_X(X=mean_X, means=mean_y, vars=var_y, delta=delta,verbose=True)
    assert X_new is None