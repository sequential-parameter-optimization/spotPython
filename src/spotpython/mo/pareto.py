import numpy as np


def is_pareto_efficient(costs: np.ndarray, minimize: bool = True) -> np.ndarray:
    """
    Find the Pareto-efficient points from a set of points.

    A point is Pareto-efficient if no other point exists that is better in all objectives.
    This function assumes that lower values are preferred for each objective when `minimize=True`,
    and higher values are preferred when `minimize=False`.

    Args:
        costs (np.ndarray):
            An (N,M) array-like object of points, where N is the number of points and M is the number of objectives.
        minimize (bool, optional):
            If True, the function finds Pareto-efficient points assuming
            lower values are better. If False, it assumes higher values are better.
            Defaults to True.

    Returns:
        np.ndarray:
            A boolean mask of length N, where True indicates that the corresponding point is Pareto-efficient.

    Examples:
        >>> from spotpython.mo.pareto import is_pareto_efficient
        >>> import numpy as np
        >>> costs = np.array([[1, 2], [2, 1], [3, 3], [1.5, 1.5]])
        >>> is_efficient = is_pareto_efficient(costs)
        >>> print(is_efficient)
        [ True  True False  True]
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, cost in enumerate(costs):
        if is_efficient[i]:
            if minimize:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < cost, axis=1)
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient] > cost, axis=1)
            is_efficient[i] = True
    return is_efficient
