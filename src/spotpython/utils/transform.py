import copy
import numpy as np


def transform_multby2_int(x: int) -> int:
    """Transformations for hyperparameters of type int.

    Args:
        x (int):
            input, will be multiplied by 2

    Returns:
        (int):
            The result of multiplying x by 2.

    Examples:
        >>> from spotpython.utils.transform import transform_multby2_int
        >>> transform_multby2_int(3)
        6
    """
    return int(2 * x)


def transform_power_2_int(x: int) -> int:
    """Transformations for hyperparameters of type int.

    Args:
        x (int): The exponent.

    Returns:
        (int): The result of raising 2 to the power of x.

    Examples:
        >>> from spotpython.utils.transform import transform_power_2_int
        >>> transform_power_2_int(3)
        8
    """
    return int(2**x)


def transform_power_10_int(x: int) -> int:
    """Transformations for hyperparameters of type int.
    Args:
        x (int): The exponent.

    Returns:
        (int): The result of raising 10 to the power of x.

    Examples:
        >>> from spotpython.utils.transform import transform_power_10_int
        >>> transform_power_10_int(3)
        1000
    """
    return int(10**x)


def transform_power_2(x) -> float:
    """Transformations for hyperparameters of type float.

    Args:
        x (float): The exponent.

    Returns:
        (float): The result of raising 2 to the power of x.

    Examples:
        >>> from spotpython.utils.transform import transform_power_2
        >>> transform_power_2(3)
        8
    """
    return 2**x


def transform_power_10(x) -> float:
    """Transformations for hyperparameters of type float.

    Args:
        x (float): The exponent.

    Returns:
        (float): The result of raising 10 to the power of x.

    Examples:
        >>> from spotpython.utils.transform import transform_power_10
        >>> transform_power_10(3)
        1000
    """
    return 10**x


def transform_none_to_None(x) -> str:
    """
    Transformations for hyperparameters of type None.
    If the input is "none", the output is None.

    Args:
        x (str): The string to transform.

    Returns:
        (str): The transformed string.

    Examples:
        >>> from spotpython.utils.transform import transform_none_to_None
        >>> transform_none_to_None("none")
        None

    Note:
        Needed for sklearn.linear_model.LogisticRegression
    """
    if x == "none":
        return None
    else:
        return x


def transform_power(base: int, x: int, as_int: bool = False) -> float:
    """
    Raises a given base to the power of x.

    Args:
        base (int):
            The base to raise to the power of x.
        x (int):
            The exponent.
        as_int (bool):
            If True, returns the result as an integer.

    Returns:
        (float):
            The result of raising the base to the power of x.

    Examples:
        >>> from spotpython.utils.transform import transform_power
        >>> transform_power(2, 3)
        8
    """
    result = base**x
    if as_int:
        result = int(result)
    return result


def transform_hyper_parameter_values(fun_control, hyper_parameter_values):
    """
    Transform the values of the hyperparameters according to the transform function specified in fun_control
    if the hyperparameter is of type "int", or "float" or "num".
    Let fun_control = {"core_model_hyper_dict":{ "leaf_prediction":
    { "levels": ["mean", "model", "adaptive"], "type": "factor", "default": "mean", "core_model_parameter_type": "str"},
    "max_depth": { "type": "int", "default": 20, "transform": "transform_power_2", "lower": 2, "upper": 20}}}
    and v = {'max_depth': 20,'leaf_prediction': 'mean'} and def transform_power_2(x): return 2**x.
    The function takes fun_control and v as input and returns a dictionary with the same structure as v.
    The function transforms the values of the hyperparameters according to the transform function
    specified in fun_control if the hyperparameter is of type "int", or "float" or "num".
    For example, transform_hyper_parameter_values(fun_control, v) returns
     {'max_depth': 1048576, 'leaf_prediction': 'mean'}.

    Args:
        fun_control (dict):
            A dictionary containing the information about the core model and the hyperparameters.
        hyper_parameter_values (dict):
            A dictionary containing the values of the hyperparameters.

    Returns:
        (dict):
            A dictionary containing the values of the hyperparameters.

    Examples:
            >>> from spotpython.utils.transform import transform_hyper_parameter_values
                fun_control = {
                    "core_model_hyper_dict": {
                        "leaf_prediction": {
                                "type": "factor",
                                "transform": "None",
                                "default": "mean",
                                "levels": ["mean", "model", "adaptive"],
                                "core_model_parameter_type": "str"
                                            },
                        "max_depth": {
                                "type": "int",
                                "default": 20,
                                "transform": "transform_power_2",
                                "lower": 2,
                                "upper": 20}
                            }
                    }
                hyper_parameter_values = {
                        'max_depth': 2,
                        'leaf_prediction': 'mean'}
                transform_hyper_parameter_values(fun_control, hyper_parameter_values)
                    {'max_depth': 4, 'leaf_prediction': 'mean'}
                fun_control = {
                    "core_model_hyper_dict": {
                        "l1": {
                            "type": "int",
                            "default": 3,
                            "transform": "transform_power_2_int",
                            "lower": 3,
                            "upper": 8
                        },
                        "epochs": {
                            "type": "int",
                            "default": 4,
                            "transform": "transform_power_2_int",
                            "lower": 4,
                            "upper": 9
                        },
                        "batch_size": {
                            "type": "int",
                            "default": 4,
                            "transform": "transform_power_2_int",
                            "lower": 1,
                            "upper": 4
                        },
                        "act_fn": {
                            "levels": [
                                "Sigmoid",
                                "Tanh",
                                "ReLU",
                                "LeakyReLU",
                                "ELU",
                                "Swish"
                            ],
                            "type": "factor",
                            "default": "ReLU",
                            "transform": "None",
                            "class_name": "spotpython.torch.activation",
                            "core_model_parameter_type": "instance()",
                            "lower": 0,
                            "upper": 5
                        },
                        "optimizer": {
                            "levels": [
                                "Adadelta",
                                "Adagrad",
                                "Adam",
                                "AdamW",
                                "SparseAdam",
                                "Adamax",
                                "ASGD",
                                "NAdam",
                                "RAdam",
                                "RMSprop",
                                "Rprop",
                                "SGD"
                            ],
                            "type": "factor",
                            "default": "SGD",
                            "transform": "None",
                            "class_name": "torch.optim",
                            "core_model_parameter_type": "str",
                            "lower": 0,
                            "upper": 11
                        },
                        "dropout_prob": {
                            "type": "float",
                            "default": 0.01,
                            "transform": "None",
                            "lower": 0.0,
                            "upper": 0.25
                        },
                        "lr_mult": {
                            "type": "float",
                            "default": 1.0,
                            "transform": "None",
                            "lower": 0.1,
                            "upper": 10.0
                        },
                        "patience": {
                            "type": "int",
                            "default": 2,
                            "transform": "transform_power_2_int",
                            "lower": 2,
                            "upper": 6
                        },
                        "batch_norm": {
                            "levels": [
                                0,
                                1
                            ],
                            "type": "factor",
                            "default": 0,
                            "transform": "None",
                            "core_model_parameter_type": "bool",
                            "lower": 0,
                            "upper": 1
                        },
                        "initialization": {
                            "levels": [
                                "Default",
                                "kaiming_uniform",
                                "kaiming_normal",
                                "xavier_uniform",
                                "xavier_normal"
                            ],
                            "type": "factor",
                            "default": "Default",
                            "transform": "None",
                            "core_model_parameter_type": "str",
                            "lower": 0,
                            "upper": 4
                        }
                    }
                }
                hyper_parameter_values = {
                        'l1': 2,
                        'epochs': 3,
                        'batch_size': 4,
                        'act_fn': 'ReLU',
                        'optimizer': 'SGD',
                        'dropout_prob': 0.01,
                        'lr_mult': 1.0,
                        'patience': 3,
                        'batch_norm': 0,
                        'initialization': 'Default',
                    }
                transform_hyper_parameter_values(fun_control, hyper_parameter_values)
                    {'l1': 4,
                    'epochs': 8,
                    'batch_size': 16,
                    'act_fn': 'ReLU',
                    'optimizer': 'SGD',
                    'dropout_prob': 0.01,
                    'lr_mult': 1.0,
                    'patience': 8,
                    'batch_norm': 0,
                    'initialization': 'Default'}
    """
    hyper_parameter_values = copy.deepcopy(hyper_parameter_values)
    for key, value in hyper_parameter_values.items():
        if fun_control["core_model_hyper_dict"][key]["type"] in ["int", "float", "num", "factor"] and fun_control["core_model_hyper_dict"][key]["transform"] != "None":
            hyper_parameter_values[key] = eval(fun_control["core_model_hyper_dict"][key]["transform"])(value)
    return hyper_parameter_values


def scale(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Sample scaling from unit hypercube to different bounds. Converts a sample from `[0, 1)` to `[a, b)`.
    The following transformation is used:
    `(b - a) * X + a`

    Note:
        equal lower and upper bounds are feasible.

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
        >>> from spotpython.utils.transform import scale
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
    # Vectorized scaling operation
    X = (upper - lower) * X + lower
    # Handling case where lower == upper
    X[:, lower == upper] = lower[lower == upper]
    return X


def nat_to_cod_X(X, cod_type):
    """
    Compute coded X-values from natural (physical or real world) units based on the
    setting of the `cod_type` attribute. If `cod_type` is "norm", the values are
    normalized to [0,1]. If `cod_type` is "std", the values are standardized.
    Otherwise, the values are not modified.

    Args:
        X (np.array): The input array.
        cod_type (str): The type of coding ("norm", "std", or other).

    Returns:
        cod_X (np.array): The coded X-values.
        min_X (np.array): The minimum values of X.
        max_X (np.array): The maximum values of X.
        mean_X (np.array): The mean values of X.
        std_X (np.array): The standard deviation of X.
    """
    min_X = np.min(X, axis=0)
    max_X = np.max(X, axis=0)
    mean_X = np.mean(X, axis=0)
    # make std_X array similar to mean_X array
    std_X = np.zeros_like(mean_X)
    X_copy = copy.deepcopy(X)
    # k is the number of columns in X, i.e., the dimension of the input space.
    k = X.shape[1]
    if cod_type == "norm":
        # Normalize X to [0,1] column-wise. If the range is zero, set the value to 0.5.
        for i in range(k):
            if max_X[i] - min_X[i] == 0:
                X_copy[:, i] = 0.5
            else:
                X_copy[:, i] = (X_copy[:, i] - min_X[i]) / (max_X[i] - min_X[i])
        cod_X = X_copy
    elif cod_type == "std":
        # Standardize X column-wise. If the standard deviation is zero, do not divide.
        for i in range(k):
            if max_X[i] - min_X[i] == 0:
                X_copy[:, i] = 0
            else:
                std_X[i] = np.std(X_copy[:, i], ddof=1)
                X_copy[:, i] = (X_copy[:, i] - mean_X[i]) / std_X[i]
        cod_X = X_copy
    else:
        cod_X = X_copy
    return cod_X, min_X, max_X, mean_X, std_X


def nat_to_cod_y(y, cod_type) -> np.ndarray:
    """
    Compute coded y-values from natural (physical or real world) units based on the
    setting of the `cod_type` attribute. If `cod_type` is "norm", the values are
    normalized to [0,1]. If `cod_type` is "std", the values are standardized.
    Otherwise, the values are not modified.

    Args:
        y (np.array): The input array.
        cod_type (str): The type of coding ("norm", "std", or other).

    Returns:
        cod_y (np.array):
            The coded y-values.
        min_y (np.array):
            The minimum values of y.
        max_y (np.array):
            The maximum values of y.
        mean_y (np.array):
            The mean values of y.
        std_y (np.array):
            The standard deviation of y.
    """
    mean_y = np.mean(y)
    std_y = None
    min_y = min(y)
    max_y = max(y)
    y_copy = copy.deepcopy(y)
    if cod_type == "norm":
        if (max_y - min_y) != 0:
            cod_y = (y_copy - min_y) / (max_y - min_y)
        else:
            cod_y = 0.5 * np.ones_like(y_copy)
    elif cod_type == "std":
        if (max_y - min_y) != 0:
            std_y = np.std(y, ddof=1)
            cod_y = (y_copy - mean_y) / std_y
        else:
            cod_y = np.zeros_like(y_copy)
    else:
        cod_y = y_copy
    return cod_y, min_y, max_y, mean_y, std_y


def cod_to_nat_X(cod_X, cod_type, min_X=None, max_X=None, mean_X=None, std_X=None) -> np.ndarray:
    """
    Compute natural X-values from coded units based on the
    setting of the `cod_type` attribute. If `cod_type` is "norm", the values are
    de-normalized from [0,1]. If `cod_type` is "std", the values are de-standardized.
    Otherwise, the values are not modified.

    Args:
        cod_X (np.array):
            The coded X-values.
        cod_type (str):
            The type of coding ("norm", "std", or other).
        min_X (np.array):
            The minimum values of X. Defaults to None.
        max_X (np.array):
            The maximum values of X. Defaults to None.
        mean_X (np.array):
            The mean values of X. Defaults to None.
        std_X (np.array):
            The standard deviation of X. Defaults to None.

    Returns:
        X (np.array): The natural (physical or real world) X-values.
    """
    X_copy = copy.deepcopy(cod_X)
    # k is the number of columns in X, i.e., the dimension of the input space.
    k = cod_X.shape[1]
    if cod_type == "norm":
        # De-normalize X from [0,1] column-wise.
        for i in range(k):
            X_copy[:, i] = X_copy[:, i] * (max_X[i] - min_X[i]) + min_X[i]
        X = X_copy
    elif cod_type == "std":
        # De-standardize X column-wise.
        for i in range(k):
            X_copy[:, i] = X_copy[:, i] * std_X[i] + mean_X[i]
        X = X_copy
    else:
        X = X_copy
    return X


def cod_to_nat_y(cod_y, cod_type, min_y=None, max_y=None, mean_y=None, std_y=None) -> np.ndarray:
    """
    Compute natural y-values from coded units based on the
    setting of the `cod_type` attribute. If `cod_type` is "norm", the values are
    de-normalized from [0,1]. If `cod_type` is "std", the values are de-standardized.
    Otherwise, the values are not modified.

    Args:
        cod_y (np.array):
            The coded y-values.
        cod_type (str):
            The type of coding ("norm", "std", or other).
        min_y (np.array):
            The minimum values of y. Defaults to None.
        max_y (np.array):
            The maximum values of y. Defaults to None.
        mean_y (np.array):
            The mean values of y. Defaults to None.
        std_y (np.array):
            The standard deviation of y. Defaults to None.

    Returns:
        y (np.array): The natural (physical or real world) y-values.
    """
    y_copy = copy.deepcopy(cod_y)
    if cod_type == "norm":
        y = y_copy * (max_y - min_y) + min_y
    elif cod_type == "std":
        y = y_copy * std_y + mean_y
    else:
        y = y_copy
    return y
