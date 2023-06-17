import importlib
import numpy as np
import pandas as pd
from itertools import combinations


def class_for_name(module_name, class_name) -> object:
    """Returns a class for a given module and class name.

    Parameters:
        module_name (str): The name of the module.
        class_name (str): The name of the class.

    Returns:
        object: The class.

    Example:
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
    Parameters:
        df (pandas.DataFrame): The input dataframe.
        target_column (str): The name of the target column.
    Returns:
        tuple: The tuple (X, y).
    Example:
        >>> from spotPython.utils.convert import get_Xy_from_df
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        >>> X, y = get_Xy_from_df(df, "c")
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
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
        numpy.ndarray: The output array.
    Example:
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


def add_logical_columns(df, arity):
    """Adds logical columns to a DataFrame.
    Args:
        df (pandas.DataFrame): The input DataFrame.
        arity (int): The arity of the logical columns.
    Returns:
        pandas.DataFrame: The output DataFrame.
    Example:
        >>> from spotPython.utils.convert import add_logical_columns
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [True, False, True], 'B': [False, True, True], 'C': [True, True, False]})
        >>> result = add_logical_columns(df, 2)
        >>> print(result)
            A      B      C  and_A_B  or_A_B  xor_A_B  and_A_C  or_A_C  xor_A_C  and_B_C  or_B_C  xor_B_C
        0   True  False   True    False    True     True     True    True    False   False    True     True
        1  False   True   True    False    True     True    False    True     True   False    True     True
        2   True   True  False     True    True    False    False    True     True    True    True    False
    """
    # Create a copy of the input DataFrame to avoid modifying it
    result = df.copy()

    # Create empty DataFrames for the additional columns
    and_df = pd.DataFrame(index=df.index)
    or_df = pd.DataFrame(index=df.index)
    xor_df = pd.DataFrame(index=df.index)

    # Get all combinations of columns with the specified arity
    column_combinations = list(combinations(df.columns, arity))

    # Apply the logical_and, logical_or and logical_xor functions to all combinations of columns
    for cols in column_combinations:
        col_name = "_".join(cols)
        and_df[f"and_{col_name}"] = result[cols[0]]
        or_df[f"or_{col_name}"] = result[cols[0]]
        xor_df[f"xor_{col_name}"] = result[cols[0]]
        for col in cols[1:]:
            and_df[f"and_{col_name}"] &= result[col]
            or_df[f"or_{col_name}"] |= result[col]
            xor_df[f"xor_{col_name}"] ^= result[col]

    # Concatenate the input DataFrame with the additional columns
    result = pd.concat([result, and_df, or_df, xor_df], axis=1)

    return result
