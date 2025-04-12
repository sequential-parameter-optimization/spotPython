import numpy as np
import pandas as pd
from typing import Union


def get_boundaries(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the minimum and maximum values for each column in a NumPy array.

    Args:
        data (np.ndarray): A NumPy array of shape (n, k), where n is the number of rows
            and k is the number of columns.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - The first array contains the minimum values for each column, with shape (k,).
            - The second array contains the maximum values for each column, with shape (k,).

    Raises:
        ValueError: If the input array has shape (1, 0) (empty array).

    Examples:
        >>> from spotpython.design.utils import get_boundaries
        >>> import numpy as np
        >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> min_values, max_values = get_boundaries(data)
        >>> print("Minimum values:", min_values)
        Minimum values: [1 2 3]
        >>> print("Maximum values:", max_values)
        Maximum values: [7 8 9]
    """
    if data.size == 0:
        raise ValueError("Input array cannot be empty.")
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    return min_values, max_values


def generate_search_grid(x_min: np.ndarray, x_max: np.ndarray, n_points: int = 5, col_names: list = None) -> Union[pd.DataFrame, np.ndarray]:
    """
    Generates a search grid based on the minimum and maximum values of each feature.

    Args:
        x_min (np.ndarray): A NumPy array containing the minimum values for each feature.
        x_max (np.ndarray): A NumPy array containing the maximum values for each feature.
        n_points (int, optional): The number of points to generate for each feature. Defaults to 5.
        col_names (list, optional): A list of column names for the DataFrame. If None, a NumPy array is returned. Defaults to None.

    Returns:
        Union[pd.DataFrame, np.ndarray]: A Pandas DataFrame representing the search grid if col_names is provided,
            otherwise a NumPy array.

    Raises:
        ValueError: If the length of x_min and x_max are different.

    Examples:
        >>> from spotpython.design.utils import generate_search_grid
        >>> import numpy as np
        >>> x_min = np.array([0, 0, 0])
        >>> x_max = np.array([1, 1, 1])
        >>> search_grid = generate_search_grid(x_min, x_max, num_points=3)
        >>> print(search_grid)
        [[0.  0.  0. ]
         [0.  0.  0.5]
         [0.  0.  1. ]
         ...
         [1.  1.  0.5]
         [1.  1.  1. ]]

        >>> search_grid = generate_search_grid(x_min, x_max, num_points=3, col_names=['feature_0', 'feature_1', 'feature_2'])
        >>> print(search_grid)
           feature_0  feature_1  feature_2
        0        0.0      0.00      0.00
        1        0.0      0.00      0.50
        2        0.0      0.00      1.00
        3        0.0      0.50      0.00
        4        0.0      0.50      0.50
        ..       ...      ...      ...
        22       1.0      1.00      0.50
        23       1.0      1.00      1.00

        [27 rows x 3 columns]
    """
    if len(x_min) != len(x_max):
        raise ValueError("x_min and x_max must have the same length.")

    num_features = len(x_min)
    # Create linspace for each dimension
    ranges = [np.linspace(x_min[i], x_max[i], n_points) for i in range(num_features)]

    # Use meshgrid to create all combinations
    # The maximum number of inputs for np.broadcast is 32
    if num_features > 30:
        raise ValueError("Too many features for meshgrid. Maximum 30 features are supported.")
    mesh = np.meshgrid(*ranges, indexing="ij")

    # Reshape the meshgrid output to a list of points
    points = np.array([m.ravel() for m in mesh]).T

    if col_names:
        # Create a Pandas DataFrame from the points
        if len(col_names) != num_features:
            raise ValueError("The number of column names must match the number of features.")
        search_grid = pd.DataFrame(points, columns=col_names)
        return search_grid
    else:
        return points


def map_to_original_scale(X_search: Union[pd.DataFrame, np.ndarray], x_min: np.ndarray, x_max: np.ndarray) -> Union[pd.DataFrame, np.ndarray]:
    """
    Maps the values in X_search from the range [0, 1] to the original scale defined by x_min and x_max.

    Args:
        X_search (Union[pd.DataFrame, np.ndarray]): A Pandas DataFrame or NumPy array containing the search points in the range [0, 1].
        x_min (np.ndarray): A NumPy array containing the minimum values for each feature in the original scale.
        x_max (np.ndarray): A NumPy array containing the maximum values for each feature in the original scale.

    Returns:
        Union[pd.DataFrame, np.ndarray]: A Pandas DataFrame or NumPy array with the values mapped to the original scale.

    Examples:
        >>> from spotpython.design.utils import map_to_original_scale
        >>> import numpy as np
        >>> import pandas as pd
        >>> X_search = pd.DataFrame([[0.5, 0.5], [0.25, 0.75]], columns=['x', 'y'])
        >>> x_min = np.array([0, 0])
        >>> x_max = np.array([10, 20])
        >>> X_search_scaled = map_to_original_scale(X_search, x_min, x_max)
        >>> print(X_search_scaled)
              x     y
        0   5.0  10.0
        1   2.5  15.0
    """
    if not isinstance(X_search, (pd.DataFrame, np.ndarray)):
        raise TypeError("X_search must be a Pandas DataFrame or a NumPy array.")

    # if x_min or x_max are not numpy arrays, convert them to numpy arrays
    if not isinstance(x_min, np.ndarray):
        x_min = np.array(x_min)
    if not isinstance(x_max, np.ndarray):
        x_max = np.array(x_max)

    if len(x_min) != X_search.shape[1]:
        raise IndexError(f"x_min and X_search must have the same number of columns. x_min has {len(x_min)} columns and X_search has {X_search.shape[1]} columns.")
    if len(x_max) != X_search.shape[1]:
        raise IndexError(f"x_max and X_search must have the same number of columns. x_max has {len(x_max)} columns and X_search has {X_search.shape[1]} columns.")

    if isinstance(X_search, pd.DataFrame):
        X_search_scaled = X_search.copy()  # Create a copy to avoid modifying the original DataFrame
        for i, col in enumerate(X_search.columns):
            X_search_scaled.loc[:, col] = X_search[col] * (x_max[i] - x_min[i]) + x_min[i]
        return X_search_scaled
    elif isinstance(X_search, np.ndarray):
        X_search_scaled = X_search.copy()  # Create a copy to avoid modifying the original array
        for i in range(X_search.shape[1]):
            X_search_scaled[:, i] = X_search[:, i] * (x_max[i] - x_min[i]) + x_min[i]
        return X_search_scaled
