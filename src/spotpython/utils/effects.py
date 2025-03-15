import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def randorient(k, p, xi):
    # Step length
    Delta = xi / (p - 1)

    m = k + 1

    # A truncated p-level grid in one dimension
    xs = np.arange(0, 1, Delta)
    xsl = len(xs)

    # Basic sampling matrix
    B = np.vstack((np.zeros((1, k)), np.tril(np.ones((k, k)))))

    # Randomization

    # Matrix with +1s and -1s on the diagonal with equal probability
    Dstar = np.diag(2 * np.round(np.random.rand(k)) - 1)

    # Random base value
    xstar = xs[(np.random.rand(k) * xsl).astype(int)]

    # Permutation matrix
    Pstar = np.zeros((k, k))
    rp = np.random.permutation(k)
    for i in range(k):
        Pstar[i, rp[i]] = 1

    # A random orientation of the sampling matrix
    Bstar = (np.ones((m, 1)) @ xstar.reshape(1, -1) + (Delta / 2) * ((2 * B - np.ones((m, k))) @ Dstar + np.ones((m, k)))) @ Pstar

    return Bstar


def screeningplan(k, p, xi, r):
    # Empty list to accumulate screening plan rows
    X = []

    for i in range(r):
        X.append(randorient(k, p, xi))

    # Concatenate list of arrays into a single array
    X = np.vstack(X)

    return X


def screening(X, fun, xi, p, labels, range=None, print=False) -> pd.DataFrame:
    """Generates a DataFrame with elementary effect screening metrics.

    This function calculates the mean and standard deviation of the
    elementary effects for a given set of design variables and returns
    the results as a Pandas DataFrame.

    Args:
        X (np.ndarray): The screening plan matrix, typically structured
            within a [0,1]^k box.
        fun (object): The objective function to evaluate at each
            design point in the screening plan.
        xi (float): The elementary effect step length factor.
        p (int): Number of discrete levels along each dimension.
        labels (list of str): A list of variable names corresponding to
            the design variables.
        range (np.ndarray): A 2xk matrix where the first row contains
            lower bounds and the second row contains upper bounds for
            each variable.

    Returns:
        pd.DataFrame: A DataFrame containing three columns:
            - 'varname': The name of each variable.
            - 'mean': The mean of the elementary effects for each variable.
            - 'sd': The standard deviation of the elementary effects for
            each variable.

    Examples:
        >>> import numpy as np
        >>> from spotpython.fun.objectivefunctions import Analytical
        >>> from spotpython.utils.effects import screening
        >>>
        >>> # Create a small test input with shape (n, 10)
        >>> X_test = np.array([
        ...     [0.0]*10,
        ...     [1.0]*10
        ... ])
        >>> fun = Analytical()
        >>> labels = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
        >>> result = screening(X_test, fun.fun_wingwt, np.array([[0]*10, [1]*10]), 0.1, 3, labels)
        >>> print
    """
    # Determine the number of design variables (k)
    k = X.shape[1]
    # Determine the number of repetitions (r)
    r = X.shape[0] // (k + 1)

    # Scale each design point to the given range and evaluate the objective function
    t = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if range is not None:
            X[i, :] = range[0, :] + X[i, :] * (range[1, :] - range[0, :])
        t[i] = fun(X[i, :])

    # Calculate the elementary effects
    F = np.zeros((k, r))
    for i in range(r):
        for j in range(i * (k + 1), i * (k + 1) + k):
            index = np.where(X[j, :] - X[j + 1, :] != 0)[0][0]
            F[index, i] = (t[j + 1] - t[j]) / (xi / (p - 1))

    # Compute statistical measures
    ssd = np.std(F, axis=1)
    sm = np.abs(np.mean(F, axis=1))

    if print:
        # sort the variables by decreasing mean
        idx = np.argsort(-sm)
        labels = [labels[i] for i in idx]
        sm = sm[idx]
        ssd = ssd[idx]
        df = pd.DataFrame({"varname": labels, "mean": sm, "sd": ssd})

        return df
    else:
        # Generate plot
        plt.figure()

        for i in range(k):
            plt.text(sm[i], ssd[i], labels[i], fontsize=10)

        plt.axis([min(sm), 1.1 * max(sm), min(ssd), 1.1 * max(ssd)])
        plt.xlabel("Sample means")
        plt.ylabel("Sample standard deviations")
        plt.gca().set_xlabel("Sample means")
        plt.gca().set_ylabel("Sample standard deviations")
        plt.gca().tick_params(labelsize=10)
        plt.grid(True)
        plt.show()
