from typing import Tuple
from typing import List

import numpy as np
import pandas as pd


def selectNew(A: np.ndarray, X: np.ndarray, tolerance: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select rows from A that are not in X.

    Args:
        A (numpy.ndarray): A array with new values
        X (numpy.ndarray): X array with known values
        tolerance (float): tolerance value for comparison

    Returns:
        (numpy.ndarray): array with unknown (new) values
        (numpy.ndarray): array with `True` if value is new, otherwise `False`.

    Examples:
    >>> from spotpython.utils.compare import selectNew
        import numpy as np
        A = np.array([[1,2,3],[4,5,6]])
        X = np.array([[1,2,3],[4,5,6]])
        B, ind  = selectNew(A, X)
        assert B.shape[0] == 0
        assert np.equal(ind, np.array([False, False])).all()
    >>> from spotpython.utils.compare import selectNew
        A = np.array([[1,2,3],[4,5,7]])
        X = np.array([[1,2,3],[4,5,6]])
        B, ind  = selectNew(A, X)
        assert B.shape[0] == 1
        assert np.equal(ind, np.array([False, True])).all()
    """
    B = np.abs(A[:, None] - X)
    ind = np.any(np.all(B <= tolerance, axis=2), axis=1)
    return A[~ind], ~ind


def find_equal_in_lists(a: List[int], b: List[int]) -> List[int]:
    """Find equal values in two lists.

    Args:
        a (list): list with a values
        b (list): list with b values

    Returns:
        list: list with 1 if equal, otherwise 0

    Examples:
        >>> from spotpython.utils.compare import find_equal_in_lists
            a = [1, 2, 3, 4, 5]
            b = [1, 2, 3, 4, 5]
            find_equal_in_lists(a, b)
            [1, 1, 1, 1, 1]
    """
    equal = [1 if a[i] == b[i] else 0 for i in range(len(a))]
    return equal


def check_identical_columns_and_rows(df, remove=False, verbosity=1) -> pd.DataFrame:
    """
    Checks for exact identical columns and rows in the DataFrame.

    Note:
        This is an efficient method for checking exact duplicates in a DataFrame.
        If checks with tolerance are needed, use `check_identical_columns_and_rows_with_tol()`.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        remove (bool): Whether to remove duplicate columns/rows.
        verbosity (int): Level of verbosity; 0 for no output, 1 for standard messages.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed if specified.

    Example:
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [4, 5, 6]})
        >>> check_identical_columns_and_rows(df, "Example DataFrame", remove=False, verbosity=1)
        Identical columns in Example DataFrame:
        ['A', 'B']
    """
    # Check for exact identical columns
    col_mask = df.T.duplicated(keep="first")
    if col_mask.any() and verbosity > 0:
        print(list(df.columns[col_mask]))

    if remove:
        df = df.loc[:, ~col_mask]

    # Check for exact identical rows
    row_mask = df.duplicated(keep="first")
    if row_mask.any() and verbosity > 0:
        print(list(df.index[row_mask]))

    if remove:
        df = df.loc[~row_mask]

    return df


def check_identical_columns_and_rows_with_tol(df, tolerance, remove=False, verbosity=1) -> pd.DataFrame:
    """
    Checks for identical columns and rows within a given tolerance.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        tolerance (float): The tolerance for checking equivalence.
        remove (bool): Whether to remove duplicates found within the tolerance.
        verbosity (int): Level of verbosity; 0 for no output, 1 for standard messages.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed if specified.

    Example:
        >>> df = pd.DataFrame({"A": [1, 1, 3], "B": [1, 1.01, 3], "C": [4, 5, 6]})
        >>> check_identical_columns_and_rows_with_tol(df, "Example DataFrame", tolerance=0.05, remove=False, verbosity=1)
        Identical columns within tolerance in Example DataFrame:
        ('A', 'B')
    """

    # Function to compare rows/columns with tolerance
    def is_identical_with_tolerance(series1, series2, tol):
        return np.allclose(series1, series2, atol=tol)

    # Check for identical columns within tolerance
    identical_columns = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            if is_identical_with_tolerance(df.iloc[:, i], df.iloc[:, j], tolerance):
                identical_columns.append((df.columns[i], df.columns[j]))

    if identical_columns and verbosity > 0:
        for col_pair in identical_columns:
            print(col_pair)

    if remove:
        df = df.drop(columns=[col_pair[1] for col_pair in identical_columns])

    # Check for identical rows within tolerance
    identical_rows = []
    for i in range(len(df.index)):
        for j in range(i + 1, len(df.index)):
            if is_identical_with_tolerance(df.iloc[i, :], df.iloc[j, :], tolerance):
                identical_rows.append((df.index[i], df.index[j]))

    if identical_rows and verbosity > 0:
        for row_pair in identical_rows:
            print(row_pair)

    if remove:
        df = df.drop(index=[row_pair[1] for row_pair in identical_rows])

    return df
