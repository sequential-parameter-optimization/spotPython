import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def aggregate_mean_var_base(X, y, sort=False, var_empirical=False) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Aggregate array to mean and variance per group.
    Note: The empirical variance might result in nan values.
    Therefore, the theoretical variance is calculated by default.

    Args:
        X (numpy.ndarray):
            X array, shape `(n, k)`.
        y (numpy.ndarray):
            values, shape `(n,)`.
        sort (bool):
            Whether to sort the resulting DataFrame by the group keys.
        var_empirical (bool):
            Whether to calculate the empirical variance. Default is False, which
            avoids nan values in the variance calculation.

    Returns:
        (numpy.ndarray):
            aggregated `X` values, shape `(n-m, k)`, if `m` duplicates in `X`.
        (numpy.ndarray):
            aggregated (mean per group) `y` values, shape `(1,)`, if `m` duplicates in `X`.
        (numpy.ndarray):
            aggregated (variance per group) `y` values, shape `(1,)`, if `m` duplicates in `X`.

    Examples:
        >>> from spotpython.utils.aggregate import aggregate_mean_var
            import numpy as np
            X = np.array([[1, 2], [3, 4], [1, 2]])
            y = np.array([1, 2, 3])
            X_agg, y_mean, y_var = aggregate_mean_var(X, y)
            print(X_agg)
            [[1. 2.] [3. 4.]]
            print(y_mean)
            [2. 2.]
            print(y_var)
            [1 0]
        # Empirical variance might result in nan values, see the example below
        >>> X_agg, y_mean, y_var = aggregate_mean_var(X, y, var_empirical=True)
            print(X_agg)
            print(y_mean)
            print(y_var)
            [[1 2]
            [3 4]]
            [2. 2.]
            [ 2. nan]
        >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1,2]])
            y = np.array([1, 2, 3, 4, 5])
            X_agg, y_mean, y_var = aggregate_mean_var(X, y, var_empirical=True)
            print(X_agg)
            print(y_mean)
            print(y_var)
            [[1 2]
            [3 4]]
            [3. 3.]
            [4. 2.]
        >>> X_1 = np.ones((2, 3))
            y_1 = np.sum(X_1, axis=1)
            y_2 = 2 * y_1
            X_2 = np.append(X_1, 2 * X_1, axis=0)
            X = np.append(X_2, X_1, axis=0)
            y = np.append(y_1, y_2, axis=0)
            y = np.append(y, y_2, axis=0)
            print(X)
            print(y)
            Z = aggregate_mean_var(X, y, var_empirical=True)
            print(Z)
            [[1. 1. 1.]
            [1. 1. 1.]
            [2. 2. 2.]
            [2. 2. 2.]
            [1. 1. 1.]
            [1. 1. 1.]]
            [3. 3. 6. 6. 6. 6.]
            (array([[1., 1., 1.],
                [2., 2., 2.]]), array([4.5, 6. ]), array([3., 0.]))
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        # convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)

    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of rows in X must match the length of y.")

    # Create a DataFrame from X with y as the group target
    df = pd.DataFrame(X)
    df["y"] = y

    # Define a custom function to calculate the theoretical variance
    def theoretical_var(group):
        n = len(group)
        if n == 0:
            return np.nan
        mean = group.mean()
        return ((group - mean) ** 2).sum() / n

    if var_empirical:
        # Group by all X columns, calculating the mean and empirical variance of y for each group
        grouped = df.groupby(list(df.columns[:-1]), as_index=False, sort=sort).agg(y_mean=pd.NamedAgg(column="y", aggfunc="mean"), y_var=pd.NamedAgg(column="y", aggfunc="var"))
    else:
        # Group by all X columns, calculating the mean and theoretical variance of y for each group
        grouped = df.groupby(list(df.columns[:-1]), as_index=False, sort=sort).agg(y_mean=pd.NamedAgg(column="y", aggfunc="mean"), y_var=pd.NamedAgg(column="y", aggfunc=theoretical_var))

    # Extract mean and variance results from the grouped DataFrame
    y_mean = grouped["y_mean"].to_numpy()
    y_var = grouped["y_var"].to_numpy()

    # Extract the unique X values
    X_agg = grouped.iloc[:, :-2].to_numpy()

    return X_agg, y_mean, y_var


def aggregate_mean_var(X: np.ndarray, y: np.ndarray, sort: bool = False, var_empirical: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure NumPy implementation of aggregate_mean_var for better performance.

    This version avoids pandas overhead and may be faster for large datasets.

    Args:
        X (np.ndarray): Feature array, shape (n, k).
        y (np.ndarray): Target values, shape (n,).
        sort (bool): Whether to sort the results by the group keys. Default is False.
        var_empirical (bool): Whether to calculate the empirical (sample) variance.
                             Default is False, which uses theoretical (population) variance.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - X_agg: Aggregated unique X values, shape (n_groups, k)
            - y_mean: Mean of y values per group, shape (n_groups,)
            - y_var: Variance of y values per group, shape (n_groups,)

    Raises:
        ValueError: If input arrays have incompatible shapes or dimensions.

    Examples:
        >>> from spotpython.utils.aggregate import aggregate_mean_var
            import numpy as np
            X = np.array([[1, 2], [3, 4], [1, 2]])
            y = np.array([1, 2, 3])
            X_agg, y_mean, y_var = aggregate_mean_var(X, y)
            print(X_agg)
            [[1. 2.] [3. 4.]]
            print(y_mean)
            [2. 2.]
            print(y_var)
            [1 0]
        # Empirical variance might result in nan values, see the example below
        >>> X_agg, y_mean, y_var = aggregate_mean_var(X, y, var_empirical=True)
            print(X_agg)
            print(y_mean)
            print(y_var)
            [[1 2]
            [3 4]]
            [2. 2.]
            [ 2. nan]
        >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1,2]])
            y = np.array([1, 2, 3, 4, 5])
            X_agg, y_mean, y_var = aggregate_mean_var(X, y, var_empirical=True)
            print(X_agg)
            print(y_mean)
            print(y_var)
            [[1 2]
            [3 4]]
            [3. 3.]
            [4. 2.]
        >>> X_1 = np.ones((2, 3))
            y_1 = np.sum(X_1, axis=1)
            y_2 = 2 * y_1
            X_2 = np.append(X_1, 2 * X_1, axis=0)
            X = np.append(X_2, X_1, axis=0)
            y = np.append(y_1, y_2, axis=0)
            y = np.append(y, y_2, axis=0)
            print(X)
            print(y)
            Z = aggregate_mean_var(X, y, var_empirical=True)
            print(Z)
            [[1. 1. 1.]
            [1. 1. 1.]
            [2. 2. 2.]
            [2. 2. 2.]
            [1. 1. 1.]
            [1. 1. 1.]]
            [3. 3. 6. 6. 6. 6.]
            (array([[1., 1., 1.],
                [2., 2., 2.]]), array([4.5, 6. ]), array([3., 0.]))
    """
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Invalid input shapes")

    if X.shape[0] == 0:
        return np.empty((0, X.shape[1])), np.array([]), np.array([])

    # Use lexsort for stable sorting if requested
    if sort:
        sort_idx = np.lexsort([X[:, i] for i in range(X.shape[1] - 1, -1, -1)])
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
    else:
        X_sorted = X
        y_sorted = y

    # Find unique rows and group indices
    _, unique_idx, inverse_idx = np.unique(X_sorted, axis=0, return_index=True, return_inverse=True)

    X_agg = X_sorted[unique_idx]

    # Calculate mean and variance for each group
    n_groups = len(unique_idx)
    y_mean = np.zeros(n_groups)
    y_var = np.zeros(n_groups)

    for i in range(n_groups):
        group_mask = inverse_idx == i
        group_y = y_sorted[group_mask]

        y_mean[i] = np.mean(group_y)

        if var_empirical:
            y_var[i] = np.var(group_y, ddof=1) if len(group_y) > 1 else np.nan
        else:
            y_var[i] = np.var(group_y, ddof=0)

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
            [[ 5  6]
            [ 9 10]
            [ 1  2]]
            print(selected_y)
            [3 5 1]

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
