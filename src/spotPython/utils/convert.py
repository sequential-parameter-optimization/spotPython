import importlib
import numpy as np
import pandas as pd
from itertools import combinations


def class_for_name(module_name, class_name) -> object:
    """Returns a class for a given module and class name.

    Args:
        module_name (str): The name of the module.
        class_name (str): The name of the class.

    Returns:
        object: The class.

    Examples:
        >>> from spotPython.utils.convert import class_for_name
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
        >>> from spotPython.utils.convert import get_Xy_from_df
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
        >>> from spotPython.utils.convert import series_to_array
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
