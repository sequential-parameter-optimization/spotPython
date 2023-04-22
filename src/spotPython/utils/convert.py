import importlib


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
