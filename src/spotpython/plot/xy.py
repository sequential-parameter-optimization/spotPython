import matplotlib.pyplot as plt
import numpy as np


def plot_y_vs_X(X, y, nrows=5, ncols=2, figsize=(30, 20), ylabel="y", feature_names=None):
    """
    Plots y versus each feature in X.

    Args:
        X (ndarray):
            2D array of input features.
        y (ndarray):
            1D array of target values.
        nrows (int, optional):
            Number of rows in the subplot grid. Defaults to 5.
        ncols (int, optional):
            Number of columns in the subplot grid. Defaults to 2.
        figsize (tuple, optional):
            Size of the entire figure. Defaults to (30, 20).
        ylabel (str, optional):
            Label for the y-axis. Defaults to 'y'.
        feature_names (list of str, optional):
            List of feature names. Defaults to None. If None, generates feature names as x0, x1, etc.

    Examples:
        >>> from sklearn.datasets import load_diabetes
        >>> from spotpython.plot.xy import plot_y_vs_X
        >>> data = load_diabetes()
        >>> X, y = data.data, data.target
        >>> plot_y_vs_X(X, y, nrows=5, ncols=2, figsize=(20, 15))
    """
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, (ax, col) in enumerate(zip(axs.flat, feature_names)):
        x = X[:, i]
        pf = np.polyfit(x, y, 1)
        p = np.poly1d(pf)

        ax.plot(x, y, "o")
        ax.plot(x, p(x), "r--")

        ax.set_title(col + " " + ylabel)
        ax.set_xlabel(col)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()
