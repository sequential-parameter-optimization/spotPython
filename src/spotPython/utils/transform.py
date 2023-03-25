import copy


def scale(X, lower, upper):
    """
    Sample scaling from unit hypercube to different bounds.

    Converts a sample from `[0, 1)` to `[a, b)`.
    Note: equal lower and upper bounds are feasible.
    The following transformation is used:

    `(b - a) * X + a`

    Args:
    X (array):
        Sample to scale.
    lower (array):
        lower bound of transformed data.
    upper (array):
        upper bounds of transformed data.

    Returns:
    (array):
        Scaled sample.

    Examples:
    Transform three samples in the unit hypercube to (lower, upper) bounds:

    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> from spotPython.utils.transform import scale
    >>> lower = np.array([6, 0])
    >>> upper = np.array([6, 5])
    >>> sample = np.array([[0.5 , 0.75],
    >>>             [0.5 , 0.5],
    >>>             [0.75, 0.25]])
    >>> scale(sample, lower, upper)

    """
    # Checking that X is within (0,1) interval
    if (X.max() > 1.0) or (X.min() < 0.0):
        raise ValueError("Sample is not in unit hypercube")

    for i in range(X.shape[1]):
        if lower[i] == upper[i]:
            X[:, i] = lower[i]
        else:
            X[:, i] = X[:, i] * (upper[i] - lower[i]) + lower[i]
    return X


def transform_power_2_int(x: int) -> int:
    return int(2**x)


def transform_power_10_int(x: int) -> int:
    return int(10**x)


def transform_power_2(x):
    return 2**x


def transform_hyper_parameter_values(fun_control, hyper_parameter_values):
    """
    Transform the values of the hyperparameters according to the transform function specified in f_c if the hyperparameter is of type "int", or "float" or "num".
    Let f_c = {"core_model_hyper_dict":{ "leaf_prediction": { "levels": ["mean", "model", "adaptive"], "type": "factor", "default": "mean", "core_model_parameter_type": "str"},  "max_depth": { "type": "int", "default": 20, "transform": "transform_power_2", "lower": 2, "upper": 20}}} and v = {'max_depth': 20,'leaf_prediction': 'mean'} and def transform_power_2(x): return 2**x.
    The function takes f_c and v as input and returns a dictionary with the same structure as v.
    The function transforms the values of the hyperparameters according to the transform function specified in f_c if the hyperparameter is of type "int", or "float" or "num".
    For example, transform_hyper_parameter_values(f_c, v) returns {'max_depth': 1048576, 'leaf_prediction': 'mean'}.
    Args:
        fun_control (dict): A dictionary containing the information about the core model and the hyperparameters.
        hyper_parameter_values (dict): A dictionary containing the values of the hyperparameters.
    Returns:
        dict: A dictionary containing the values of the hyperparameters.
    Example:
        >>> import copy
        >>> from spotPython.utils.transform import transform_hyper_parameter_values
        >>> fun_control = {"core_model_hyper_dict": {"leaf_prediction": {"levels": ["mean", "model", "adaptive"], "type": "factor", "default": "mean", "core_model_parameter_type": "str"}, "max_depth": {"type": "int", "default": 20, "transform": "transform_power_2", "lower": 2, "upper": 20}}}
        >>> hyper_parameter_values = {'max_depth': 20, 'leaf_prediction': 'mean'}
        >>> transform_hyper_parameter_values(fun_control, hyper_parameter_values)
        {'max_depth': 1048576, 'leaf_prediction': 'mean'}
    """
    hyper_parameter_values = copy.deepcopy(hyper_parameter_values)
    for key, value in hyper_parameter_values.items():
        if (
            fun_control["core_model_hyper_dict"][key]["type"] in ["int", "float", "num"]
            and fun_control["core_model_hyper_dict"][key]["transform"] != "None"
        ):
            hyper_parameter_values[key] = eval(fun_control["core_model_hyper_dict"][key]["transform"])(value)
    return hyper_parameter_values
