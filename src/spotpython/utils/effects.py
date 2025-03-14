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


def screening(X, objhandle, range_, xi, p, labels, print=False) -> pd.DataFrame:
    """
    Screening method for global sensitivity analysis.

    Args:
        X (np.ndarray): Design matrix with shape (n, k), where n is the number of design points and k is the number of design variables.
        objhandle (function): Objective function to evaluate the design points.
        range_ (np.ndarray): Array with shape (2, k) with the lower and upper bounds for each design variable.
        xi (float): Step length.
        p (int): Number of levels.
        labels (list): List with the names of the design variables.
        print (bool): If True, print the results in a table. If False, plot the results.

    Returns:
        pd.DataFrame: Table with the mean and standard deviation of the elementary effects


    """
    # Determine the number of design variables (k)
    k = X.shape[1]
    # Determine the number of repetitions (r)
    r = X.shape[0] // (k + 1)

    # Scale each design point to the given range and evaluate the objective function
    t = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        # X[i, :] = range_[0, :] + X[i, :] * (range_[1, :] - range_[0, :])
        t[i] = objhandle(X[i, :])

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
