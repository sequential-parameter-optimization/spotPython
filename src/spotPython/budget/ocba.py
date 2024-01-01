"""
OCBA: Optimal Computing Budget Allocation
"""

from spotPython.utils.aggregate import get_ranks
from numpy import int32, float64
from numpy import argpartition, repeat
from numpy import zeros, square, sqrt, full, around, array
import numpy as np


def get_ocba(means, vars, delta, verbose=False) -> array:
    """
    Optimal Computer Budget Allocation (OCBA)

    This function calculates the budget recommendations for a given set of means,
    variances, and incremental budget using the OCBA algorithm.

    References:
        [1]: Chun-Hung Chen and Loo Hay Lee: Stochastic Simulation Optimization: An Optimal Computer Budget Allocation,
        pp. 49 and pp. 215
        [2]: C.S.M Currie and T. Monks: How to choose the best setup for a system.
        A tutorial for the Simulation Workshop 2021, see:
        https://colab.research.google.com/github/TomMonks/sim-tools/blob/master/examples/sw21_tutorial.ipynb
        and
        https://github.com/TomMonks/sim-tools

    Args:
        means (numpy.array):
            An array of means.
        vars (numpy.array):
            An array of variances.
        delta (int):
            The incremental budget.
        verbose (bool):
            If True, print the results.

    Returns:
        (numpy.array): An array of budget recommendations.

    Note:
        The implementation is based on the pseudo-code in the Chen et al. (p. 49), see [1].

    Examples:
        >>> # From the Chen et al. book (p. 49):
        >>> mean_y = np.array([1,2,3,4,5])
            var_y = np.array([1,1,9,9,4])
            get_ocba(mean_y, var_y, 50)
            array([11,  9, 19,  9,  2])
    """
    if np.all(vars > 0) and (means.shape[0] > 2):
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
        if verbose:
            print("\nIn get_ocba():")
            print(f"means: {means}")
            print(f"vars: {vars}")
            print(f"delta: {delta}")
            print(f"n_designs: {n_designs}")
            print(f"Allocations: {allocations}")
            print(f"Ratios: {ratios}")
            print(f"Budget: {budget}")
            print(f"Ranks: {ranks}")
            print(f"Best: {best}")
            print(f"Second best: {second_best}")
            print(f"Select: {select}")
            print(f"Temp: {temp}")
            print(f"More runs: {more_runs}")
            print(f"Add budget: {add_budget}")
            print(f"More allocations: {more_alloc}")
        while more_alloc:
            more_alloc = False
            ratio_s = (more_runs * ratios).sum()
            add_budget[more_runs] = (budget / ratio_s) * ratios[more_runs]
            add_budget = around(add_budget).astype(int)
            mask = add_budget < allocations
            add_budget[mask] = allocations[mask]
            more_runs[mask] = 0
            if verbose:
                print("\nIn more_alloc:")
                print(f"ratio_s: {ratio_s}")
                print(f"more_runs: {more_runs}")
                print(f"add_budget: {add_budget}")
            if mask.sum() > 0:
                more_alloc = True
            if more_alloc:
                budget = allocations.sum() + delta
                budget -= (add_budget * ~more_runs).sum()
        t_budget = add_budget.sum()
        add_budget[best] += allocations.sum() + delta - t_budget
        return add_budget - allocations
    else:
        return None


def get_ocba_X(X, means, vars, delta, verbose=False) -> float64:
    """
    This function calculates the OCBA allocation and repeats the input array X along the specified axis.

    Args:
        X (numpy.ndarray): Input array to be repeated.
        means (list): List of means for each alternative.
        vars (list): List of variances for each alternative.
        delta (float): Indifference zone parameter.
        verbose (bool): If True, print the results.

    Returns:
        (numpy.ndarray): Repeated array of X along the specified axis based on the OCBA allocation.

    Examples:
        >>> X = np.array([[1,2,3],[4,5,6]])
            means = [1,2,3]
            vars = [1,1,1]
            delta = 50
            get_ocba_X(X, means, vars, delta)
            array([[1, 2, 3],
                     [1, 2, 3],
                        [1, 2, 3],
                        [4, 5, 6],
                        [4, 5, 6],
                        [4, 5, 6]])

    """
    if np.all(vars > 0) and (means.shape[0] > 2):
        o = get_ocba(means=means, vars=vars, delta=delta, verbose=verbose)
        return repeat(X, o, axis=0)
    else:
        return None
