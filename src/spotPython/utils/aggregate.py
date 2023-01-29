import pandas as pd
import numpy as np


def aggregate_mean_var(X, y, sort=False):
    """
    Aggregate array to mean.

    Args:
        X (numpy.ndarray): X array, shape `(n, k)`.
        y (numpy.ndarray): values, shape `(n,)`.

    Returns:
        (numpy.ndarray): aggregated `X` values, shape `(n-m, k)`, if `m`duplicates in `X`.
        (numpy.ndarray): aggregated (mean per group) `y` values, shape `(1,)`, if `m`duplicates in `X`.
        (numpy.ndarray): aggregated (variance per group) `y` values, shape `(1,)`, if `m`duplicates in `X`.
    """
    df = pd.DataFrame(X, dtype=pd.Float64Dtype)
    # df.columns=["X"+str(i) for i in range(df.shape[1])]
    df = df.assign(y=y)
    df_m = df.groupby(list(df.columns.difference(["y"])), as_index=False, sort=sort).mean()
    df_var = df.groupby(list(df.columns.difference(["y"])), as_index=False, sort=sort).var()
    A = df_m.to_numpy(dtype=np.float64)
    B = df_var.to_numpy()
    return np.delete(A, -1, 1), A[:, -1], B[:, -1]


def get_ranks(x):
    """
    Returns a numpy array containing ranks of numbers within an input numpy array x:

    Examples:

    get_ranks([2, 1])
    [1, 0]

    get_ranks([20, 10, 100])
    [1, 0, 2]

    Args:
        x (numpy.ndarray): numpy array

    Returns:
        (numpy.ndarray): ranks

    """
    ts = x.argsort()
    ranks = np.empty_like(ts)
    ranks[ts] = np.arange(len(x))
    return ranks
