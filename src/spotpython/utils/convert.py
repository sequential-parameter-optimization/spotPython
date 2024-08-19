import importlib
import numpy as np
import pandas as pd
from itertools import combinations
import copy


def class_for_name(module_name, class_name) -> object:
    """Returns a class for a given module and class name.

    Args:
        module_name (str): The name of the module.
        class_name (str): The name of the class.

    Returns:
        object: The class.

    Examples:
        >>> from spotpython.utils.convert import class_for_name
            from scipy.optimize import rosen
            bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
            shgo_class = class_for_name("scipy.optimize", "shgo")
            result = shgo_class(rosen, bounds)
    """
    m = importlib.import_module(module_name)
    c = getattr(m, class_name)
    return c


def get_Xy_from_df(df, target_column) -> tuple:
    """Get X and y from a dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe.
        target_column (str): The name of the target column.

    Returns:
        tuple: The tuple (X, y).

    Examples:
        >>> from spotpython.utils.convert import get_Xy_from_df
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        >>> X, y = get_Xy_from_df(df, "c")
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    # convert to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y


def find_indices(A, B):
    indices = []
    for element in A:
        index = B.index(element)
        indices.append(index)
    return indices


def series_to_array(series):
    """Converts a pandas series to a numpy array.
    Args:
        series (pandas.Series): The input series.

    Returns:
        (numpy.ndarray): The output array.

    Examples:
        >>> from spotpython.utils.convert import series_to_array
        >>> import pandas as pd
        >>> series = pd.Series([1, 2, 3])
        >>> series_to_array(series)
        array([1, 2, 3])
    """
    if isinstance(series, np.ndarray):
        return series
    else:
        return series.to_numpy()


def add_logical_columns(X, arity=2, operations=["and", "or", "xor"]):
    """Combines all features in a dataframe with each other using bitwise operations

    Args:
        X (pd.DataFrame): dataframe with features
        arity (int): the number of columns to combine at once
        operations (list of str): the operations to apply. Possible values are 'and', 'or' and 'xor'

    Returns:
        X (pd.DataFrame): dataframe with new features

    Examples:
        >>> X = pd.DataFrame({"a": [True, False, True], "b": [True, True, False], "c": [False, False, True]})
        >>> add_logical_columns(X)
            a      b      c  a_and_b  a_and_c  b_and_c  a_or_b  a_or_c  b_or_c  a_xor_b  a_xor_c  b_xor_c
        0  True   True  False     True    False    False    True    True    True    False     True     True
        1 False   True  False    False    False    False    True   False    True     True     True    False
        2  True  False   True    False     True    False    True    True    True     True    False     True

    """
    new_cols = []
    # Iterate over all combinations of columns of the given arity
    for cols in combinations(X.columns, arity):
        # Create new columns for the specified operations
        if "and" in operations:
            and_col = X[list(cols)].apply(lambda x: x.all(), axis=1)
            new_cols.append(and_col)
        if "or" in operations:
            or_col = X[list(cols)].apply(lambda x: x.any(), axis=1)
            new_cols.append(or_col)
        if "xor" in operations:
            xor_col = X[list(cols)].apply(lambda x: x.sum() % 2 == 1, axis=1)
            new_cols.append(xor_col)
    # Join all the new columns at once
    X = pd.concat([X] + new_cols, axis=1)
    return X


def map_to_True_False(value):
    """
    Map the string value to a boolean value.
    If the value is "True" or "true", return True.
    Otherwise, return False.

    Args:
        value (str):
            The string to be mapped to a boolean value.
    Returns:
        bool:
            True if the value is "True" or "true", False otherwise.

    Examples:
    >>> from spotpython.utils.convert import map_to_True_False
        map_to_True_False("True")
        True
    """
    if value.lower() == "true":
        return True
    else:
        return False


def sort_by_kth_and_return_indices(array, k) -> list:
    """Sorts an array of arrays based on the k-th values in descending order and returns
    the indices of the original array entries.

    Args:
        array (list of lists): The array to be sorted. Each sub-array should have at least
            `k+1` elements.
        k (int): The index (zero-based) of the element within each sub-array to sort by.

    Returns:
        list of int: Indices of the original array entries after sorting by the k-th value.

    Raises:
        ValueError: If the input array is empty, None, or any sub-array does not have at least
            `k+1` elements, or if k is out of bounds for any sub-array.

    Examples:
        >>> from spotpython.utils.convert import sort_by_kth_and_return_indices
            try:
                array = [['x0', 85.50983192204619], ['x1', 100.0], ['x2', 81.35712613549178]]
                k = 1  # Sort by the second element in each sub-array
                indices = sort_by_kth_and_return_indices(array, k)
                print("Indices of the sorted elements using the k-th value:", indices)
            except ValueError as error:
                print(f"Sorting failed due to: {error}")
    """
    if not array:
        return []

    # Check for improperly structured sub-arrays and that k is within bounds
    for item in array:
        if not isinstance(item, list) or len(item) <= k:
            raise ValueError("All sub-arrays must be lists with at least k+1 elements.")

    # Enumerate the array to keep track of original indices, then sort by the k-th item
    sorted_indices = [index for index, value in sorted(enumerate(array), key=lambda x: x[1][k], reverse=True)]

    return sorted_indices


def check_type(value) -> str:
    """Check the type of the input value and return the type as a string.

    Args:
        value (object): The input value.

    Returns:
        str:
            The type of the input value as a string.
            Possible values are "int", "float", "str", "bool", or None.
            Checks for numpy types as well, i.e., np.integer, np.floating, np.str_, np.bool_.

    Examples:
        >>> from spotpython.utils.convert import check_type
        >>> check_type(5)
        "int"

    """
    if isinstance(value, (int, np.integer)):
        return "int"
    elif isinstance(value, (float, np.floating)):
        return "float"
    elif isinstance(value, (str, np.str_)):
        return "str"
    elif isinstance(value, (bool, np.bool_)):
        return "bool"
    else:
        return None


def set_dataset_target_type(dataset, target="y") -> pd.DataFrame:
    """Set the target column to 0 and 1 for boolean and string values.

    Args:
        dataset (pd.DataFrame): The input dataset.
        target (str): The name of the target column. Default is "y".

    Returns:
        pd.DataFrame:
            The dataset with boolean and string target column values set to 0 and 1.

    Examples:
    >>> from spotpython.utils.convert import set_dataset_target_type
        import pandas as pd
        dataset = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "y": [True, False, True]})
        print(dataset)
        dataset = set_dataset_target_type(dataset)
        print(dataset)
            a  b  c      y
            0  1  4  7   True
            1  2  5  8  False
            2  3  6  9   True
            a  b  c  y
            0  1  4  7  1
            1  2  5  8  0
            2  3  6  9  1


    """
    val = copy.deepcopy(dataset.iloc[0, -1])
    target_type = check_type(val)
    if target_type == "bool" or target_type == "str":
        # convert the target column to 0 and 1
        dataset[target] = dataset[target].astype(int)
    return dataset
