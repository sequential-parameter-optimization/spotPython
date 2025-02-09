import pandas as pd
import numpy as np
from scipy.stats import norm, t
from numpy.linalg import pinv, inv, LinAlgError


def cov_to_cor(covariance_matrix) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Args:
        covariance_matrix (numpy.ndarray): A square matrix of covariance values.

    Returns:
        numpy.ndarray: A corresponding square matrix of correlation coefficients.

    Examples:
        >>> from spotpython.utils.stats import cov_to_cor
        >>> import numpy as np
        >>> cov_matrix = np.array([[1, 0.8], [0.8, 1]])
        >>> cov_to_cor(cov_matrix)
        array([[1. , 0.8],
               [0.8, 1. ]])
    """
    d = np.sqrt(np.diag(covariance_matrix))
    return covariance_matrix / np.outer(d, d)


def partial_correlation(x, method="pearson") -> dict:
    """Calculate the partial correlation matrix for a given data set.

    Args:
        x (pandas.DataFrame or numpy.ndarray): The data matrix with variables as columns.
        method (str): Correlation method, one of 'pearson', 'kendall', or 'spearman'.

    Returns:
        dict: A dictionary containing the partial correlation estimates, p-values,
            statistics, sample size (n), number of given parameters (gp), and method used.

    Raises:
        ValueError: If input is not a matrix-like structure or not numeric.

    References:
        1. Kim, S. ppcor: An R package for a fast calculation to semi-partial correlation coefficients.
        Commun Stat Appl Methods 22, 6 (Nov 2015), 665–674.

    Examples:
        >>> from spotpython.utils.stats import cov_to_cor
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        >>>     'A': [1, 2, 3],
        >>>     'B': [4, 5, 6],
        >>>     'C': [7, 8, 9]
        >>> })
        >>> partial_correlation(data, method='pearson')
        {'estimate': array([[ 1. , -1. ,  1. ],
                            [-1. ,  1. , -1. ],
                            [ 1. , -1. ,  1. ]]),
        'p_value': array([[0. , 0. , 0. ],
                          [0. , 0. , 0. ],
                          [0. , 0. , 0. ]]), ...
        }
    """
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    if not isinstance(x, np.ndarray):
        raise ValueError("Supply a matrix-like 'x'")
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("'x' must be numeric")

    n = x.shape[0]
    gp = x.shape[1] - 2
    cvx = np.cov(x, rowvar=False, bias=True)

    try:
        if np.linalg.det(cvx) < np.finfo(float).eps:
            icvx = pinv(cvx)
        else:
            icvx = inv(cvx)
    except LinAlgError:
        icvx = pinv(cvx)

    p_cor = -cov_to_cor(icvx)
    np.fill_diagonal(p_cor, 1)

    epsilon = 1e-10  # small value to prevent division by zero
    if method == "kendall":
        denominator = np.sqrt(2 * (2 * (n - gp) + 5) / (9 * (n - gp) * (n - 1 - gp)))
        statistic = p_cor / denominator
        p_value = 2 * norm.cdf(-np.abs(statistic))
    else:
        factor = np.sqrt((n - 2 - gp) / (1 - p_cor**2 + epsilon))
        statistic = p_cor * factor
        p_value = 2 * t.cdf(-np.abs(statistic), df=n - 2 - gp)

    np.fill_diagonal(statistic, 0)
    np.fill_diagonal(p_value, 0)

    return {"estimate": p_cor, "p_value": p_value, "statistic": statistic, "n": n, "gp": gp, "method": method}


def pairwise_partial_correlation(x, y, z, method="pearson") -> dict:
    """Calculate the pairwise partial correlation between two variables given others.

    Args:
        x (array-like): The first variable as a 1-dimensional array or list.
        y (array-like): The second variable as a 1-dimensional array or list.
        z (pandas.DataFrame): A data frame containing other conditional variables.
        method (str): Correlation method, one of 'pearson', 'kendall', or 'spearman'.

    Returns:
        dict: A dictionary with the partial correlation estimate, p-value, statistic,
            sample size (n), number of given parameters (gp), and method used.

    References:
        1. Kim, S. ppcor: An R package for a fast calculation to semi-partial correlation coefficients.
        Commun Stat Appl Methods 22, 6 (Nov 2015), 665–674.

    Examples:
        >>> from spotpython.utils.stats import pairwise_partial_correlation
        >>> import pandas as pd
        >>> x = [1, 2, 3]
        >>> y = [4, 5, 6]
        >>> z = pd.DataFrame({'C': [7, 8, 9]})
        >>> pairwise_partial_correlation(x, y, z)
        {'estimate': -1.0, 'p_value': 0.0, 'statistic': -inf, 'n': 3, 'gp': 1, 'method': 'pearson'}
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = pd.DataFrame(z)

    xyz = pd.concat([pd.Series(x), pd.Series(y), z], axis=1)

    pcor_result = partial_correlation(xyz, method=method)

    return {
        "estimate": pcor_result["estimate"][0, 1],
        "p_value": pcor_result["p_value"][0, 1],
        "statistic": pcor_result["statistic"][0, 1],
        "n": pcor_result["n"],
        "gp": pcor_result["gp"],
        "method": method,
    }


def semi_partial_correlation(x, method="pearson"):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    if not isinstance(x, np.ndarray):
        raise ValueError("Supply a matrix-like 'x'")
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("'x' must be numeric")

    n = x.shape[0]
    gp = x.shape[1] - 2
    cvx = np.cov(x, rowvar=False, bias=True)

    try:
        if np.linalg.det(cvx) < np.finfo(float).eps:
            icvx = pinv(cvx)
        else:
            icvx = inv(cvx)
    except LinAlgError:
        icvx = pinv(cvx)

    sp_cor = -cov_to_cor(icvx) / np.sqrt(np.diag(cvx)) / np.sqrt(np.abs(np.diag(icvx) - np.square(icvx.T) / np.diag(icvx)))
    np.fill_diagonal(sp_cor, 1)

    if method == "kendall":
        denominator = np.sqrt(2 * (2 * (n - gp) + 5) / (9 * (n - gp) * (n - 1 - gp)))
        statistic = sp_cor / denominator
        p_value = 2 * norm.cdf(-np.abs(statistic))
    else:
        factor = np.sqrt((n - 2 - gp) / (1 - sp_cor**2))
        statistic = sp_cor * factor
        p_value = 2 * t.cdf(-np.abs(statistic), df=n - 2 - gp)

    np.fill_diagonal(statistic, 0)
    np.fill_diagonal(p_value, 0)

    return {"estimate": sp_cor, "p_value": p_value, "statistic": statistic, "n": n, "gp": gp, "method": method}


def pairwise_semi_partial_correlation(x, y, z, method="pearson"):
    x = np.asarray(x)
    y = np.asarray(y)
    z = pd.DataFrame(z)

    xyz = pd.concat([pd.Series(x), pd.Series(y), z], axis=1)

    spcor_result = semi_partial_correlation(xyz, method=method)

    return {
        "estimate": spcor_result["estimate"][0, 1],
        "p_value": spcor_result["p_value"][0, 1],
        "statistic": spcor_result["statistic"][0, 1],
        "n": spcor_result["n"],
        "gp": spcor_result["gp"],
        "method": method,
    }
