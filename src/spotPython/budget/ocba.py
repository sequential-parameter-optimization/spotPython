"""
OCBA: Optimal Computing Budget Allocation
"""

from spotPython.utils.aggregate import get_ranks
from numpy import int32, float64
from numpy import argpartition, repeat
from numpy import zeros, square, sqrt, full, around


def get_ocba(means, vars, delta):
    """
    Optimal Computer Budget Allocation (OCBA)

    References:
        Chun-Hung Chen and Loo Hay Lee:
        Stochastic Simulation Optimization: An Optimal Computer Budget Allocation,
        pp. 49 and pp. 215

        C.S.M Currie and T. Monks:
        How to choose the best setup for a system. A tutorial for the Simulation Workshop 2021,
        see:
        https://colab.research.google.com/github/TomMonks/sim-tools/blob/master/examples/sw21_tutorial.ipynb
        and
        https://github.com/TomMonks/sim-tools

    Examples:

        From the Chen et al. book (p. 49):
        mean_y = np.array([1,2,3,4,5])
        var_y = np.array([1,1,9,9,4])
        get_ocba(mean_y, var_y, 50)

        [11  9 19  9  2]

        Args:
        means (numpy.array):
            means
        vars (numpy.array):
            variances
        delta (int):
            incremental budget

        Returns:
        (numpy.array):
            budget recommendations. `(n,)` numpy.array
    """
    n_designs = means.shape[0]
    allocations = zeros(n_designs, int32)
    ratios = zeros(n_designs, float64)
    budget = delta
    ranks = get_ranks(means)
    best, second_best = argpartition(ranks, 2)[:2]
    ratios[second_best] = 1.0
    select = [i for i in range(n_designs) if i not in [best, second_best]]
    temp = (means[best] - means[second_best]) / (means[best] - means[select])
    ratios[select] = square(temp) * (vars[select] / vars[second_best])
    select = [i for i in range(n_designs) if i not in [best]]
    temp = (square(ratios[select]) / vars[select]).sum()
    ratios[best] = sqrt(vars[best] * temp)
    more_runs = full(n_designs, True, dtype=bool)
    add_budget = zeros(n_designs, dtype=float)
    more_alloc = True
    while more_alloc:
        more_alloc = False
        ratio_s = (more_runs * ratios).sum()
        add_budget[more_runs] = (budget / ratio_s) * ratios[more_runs]
        add_budget = around(add_budget).astype(int)
        mask = add_budget < allocations
        add_budget[mask] = allocations[mask]
        more_runs[mask] = 0
        if mask.sum() > 0:
            more_alloc = True
        if more_alloc:
            budget = allocations.sum() + delta
            budget -= (add_budget * ~more_runs).sum()
    t_budget = add_budget.sum()
    add_budget[best] += allocations.sum() + delta - t_budget
    return add_budget - allocations


def get_ocba_X(X, means, vars, delta):
    o = get_ocba(means=means, vars=vars, delta=delta)
    return repeat(X, o, axis=0)
