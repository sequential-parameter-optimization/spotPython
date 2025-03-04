import numpy as np
import pandas as pd
from scipy.stats import qmc


def fried(n=50, m=6) -> pd.DataFrame:
    """
    Generate a dataset using the Friedman function.

    Args:
        n (int): Number of samples.
        m (int): Number of features (must be at least 5).

    Returns:
        pd.DataFrame: DataFrame containing the generated features, response values (Y), and true response values (Ytrue).

    Examples:
        >>> from spotpython.gp.functions import fried
        >>> data = fried(n=50, m=6)
        >>> print(data.head())
             X1        X2        X3        X4        X5        Y      Ytrue
        0  0.50  0.166667  0.333333  0.666667  0.833333  17.5728  17.5728
        1  0.25  0.500000  0.666667  0.166667  0.666667  16.8473  16.8473
        2  0.75  0.833333  0.000000  0.833333  0.333333  21.6731  21.6731
        3  0.00  0.333333  0.666667  0.500000  0.500000  14.6937  14.6937
        4  1.00  0.666667  0.333333  0.333333  0.166667  16.2804  16.2804
    """
    if m < 5:
        raise ValueError("must have at least 5 cols")

    # Generate Latin Hypercube samples
    sampler = qmc.LatinHypercube(d=m)
    X = sampler.random(n)

    # Calculate the true response values
    Ytrue = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]

    # Add noise to the response values
    Y = Ytrue + np.random.normal(0, 1, n)

    # Create a DataFrame with the generated data
    data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(m)])
    data["Y"] = Y
    data["Ytrue"] = Ytrue

    return data


def f2d(x, y=None) -> np.ndarray:
    """
    Simple 2-d test function used in Gramacy & Apley (2015).

    Args:
        x (ndarray): The x-coordinates.
        y (ndarray, optional): The y-coordinates. If None, x is assumed to be a 2D array.

    Returns:
        ndarray: The calculated z-values.
    """
    if y is None:
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            x = np.array(x).reshape(-1, 2)
        y = x[:, 1]
        x = x[:, 0]

    def g(z):
        return np.exp(-((z - 1) ** 2)) + np.exp(-0.8 * (z + 1) ** 2) - 0.05 * np.sin(8 * (z + 0.1))

    z = -g(x) * g(y)
    return z
