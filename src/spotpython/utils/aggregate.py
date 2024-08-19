import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def aggregate_mean_var(X, y, sort=False) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Aggregate array to mean.

    Args:
        X (numpy.ndarray): X array, shape `(n, k)`.
        y (numpy.ndarray): values, shape `(n,)`.
        sort (bool): Whether to sort the resulting DataFrame by the group keys.

    Returns:
        (numpy.ndarray):
            aggregated `X` values, shape `(n-m, k)`, if `m` duplicates in `X`.
        (numpy.ndarray):
            aggregated (mean per group) `y` values, shape `(1,)`, if `m` duplicates in `X`.
        (numpy.ndarray):
            aggregated (variance per group) `y` values, shape `(1,)`, if `m` duplicates in `X`.

    Examples:
        >>> X = np.array([[1, 2], [3, 4], [1, 2]])
            y = np.array([1, 2, 3])
            X_agg, y_mean, y_var = aggregate_mean_var(X, y)
            print(X_agg)
            [[1. 2.]
            [3. 4.]]
            print(y_mean)
            [2. 2.]
            print(y_var)
            [1. 0.]
    """
    # Create a DataFrame from X and y
    df = pd.DataFrame(X)
    df["y"] = y

    # Group by all columns except 'y' and calculate the mean and variance of 'y' for each group
    grouped = df.groupby(list(df.columns.difference(["y"])), as_index=False, sort=sort)
    df_mean = grouped.mean()
    df_var = grouped.var()

    # Convert the resulting DataFrames to numpy arrays
    mean_array = df_mean.to_numpy()
    var_array = df_var.to_numpy()

    # Split the resulting arrays into separate arrays for X and y
    X_agg = np.delete(mean_array, -1, 1)
    y_mean = mean_array[:, -1]
    y_var = var_array[:, -1]

    return X_agg, y_mean, y_var


def get_ranks(x):
    """
    Returns a numpy array containing ranks of numbers within an input numpy array x.

    Args:
        x (numpy.ndarray): numpy array

    Returns:
        (numpy.ndarray): ranks

    Examples:
        >>> get_ranks([2, 1])
            [1, 0]
        >>> get_ranks([20, 10, 100])
            [1, 0, 2]
    """
    ts = x.argsort()
    ranks = np.empty_like(ts)
    ranks[ts] = np.arange(len(x))
    return ranks


def select_distant_points(X, y, k):
    """
    Selects k points that are distant from each other using a clustering approach.

    Args:
        X (numpy.ndarray): X array, shape `(n, k)`.
        y (numpy.ndarray): values, shape `(n,)`.
        k (int): number of points to select.

    Returns:
        (numpy.ndarray):
            selected `X` values, shape `(k, k)`.
        (numpy.ndarray):
            selected `y` values, shape `(k,)`.

    Examples:
        >>> from spotpython.utils.aggregate import select_distant_points
            X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
            y = np.array([1, 2, 3, 4, 5])
            selected_points, selected_y = select_distant_points(X, y, 3)
            print(selected_points)
            [[1 2]
            [7 8]
            [9 10]]
            print(selected_y)
            [1 4 5]

    """
    # Perform k-means clustering to find k clusters
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
    # Find the closest point in X to each cluster center
    selected_points = np.array([X[np.argmin(np.linalg.norm(X - center, axis=1))] for center in kmeans.cluster_centers_])
    # Find indices of the selected points in the original X array
    indices = np.array([np.where(np.all(X == point, axis=1))[0][0] for point in selected_points])
    # Select the corresponding y values
    selected_y = y[indices]
    return selected_points, selected_y
