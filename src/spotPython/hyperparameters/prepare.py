import numpy as np
import copy
import json

from spotPython.utils.convert import class_for_name


def get_values(var_dict, fun_control):
    for values in iterate_dict_values(var_dict):
        values = convert_keys(values, fun_control["var_type"])
        values = get_dict_with_levels_and_types(fun_control=fun_control, v=values)
        values = transform_hyper_parameter_values(fun_control=fun_control, hyper_parameter_values=values)
        yield values


def iterate_dict_values(var_dict: dict):
    """
    This function takes a dictionary of variables as input arguments and returns an iterator that
    yields the values from the arrays in the dictionary.

    Parameters:
        var_dict (dict): A dictionary where keys are variable names and values are numpy arrays.

    Returns:
        iterator: An iterator that yields the values from the arrays in the dictionary.

    Example:
        >>> import numpy as np
        >>> var_dict = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}
        >>> for values in iterate_dict_values(var_dict):
        ...     print(values)
        {'a': 1, 'b': 2}
        {'a': 3, 'b': 4}
        {'a': 5, 'b': 6}
    """
    n = len(next(iter(var_dict.values())))
    for i in range(n):
        yield {key: value[i] for key, value in var_dict.items()}


def convert_keys(d: dict, var_type: list):
    """
    Convert values in a dictionary to integers based on a list of variable types.

    This function takes a dictionary `d` and a list of variable types `var_type` as arguments.
    For each key in the dictionary,
    if the corresponding entry in `var_type` is not equal to `"num"`,
    the value associated with that key is converted to an integer.

    Args:
        d (dict): The input dictionary.
        var_type (list): A list of variable types. If the entry is not `"num"` the corresponding
        value will be converted to the type `"int"`.

    Returns:
        dict: The modified dictionary with values converted to integers based on `var_type`.

    Example:
        >>> d = {'a': '1.1', 'b': '2', 'c': '3.1'}
        >>> var_type = ["int", "num", "int"]
        >>> convert_keys(d, var_type)
        {'a': 1, 'b': '2', 'c': 3}
    """
    keys = list(d.keys())
    for i in range(len(keys)):
        if var_type[i] not in ["num", "float"]:
            d[keys[i]] = int(d[keys[i]])
    return d


def get_dict_with_levels_and_types(fun_control, v):
    """Get dictionary with levels and types.
    The function is maps the numerical output of the hyperparameter optimization to the corresponding levels
    of the hyperparameter needed by the core model, i.e., the tuned algorithm.
    The function takes the dictionaries d and v and returns a new dictionary with the same keys as v
    but with the values of the levels of the keys from d.
    If the key value in the dictionary is 0, it takes the first value from the list,
    if it is 1, it takes the second and so on.
    If a key is not in d, it takes the key from v.
    If the core_model_parameter_type value is instance, it returns the class of the value from the module
    via getattr("class", value).
    For example,
    if d = {"HoeffdingTreeRegressor":{
        "leaf_prediction": {
            "levels": ["mean", "model", "adaptive"],
            "type": "factor",
            "default": "mean",
            "core_model_parameter_type": "str"},
        "leaf_model": {
            "levels": ["linear_model.LinearRegression", "linear_model.PARegressor", "linear_model.Perceptron"],
            "type": "factor",
            "default": "LinearRegression",
            "core_model_parameter_type": "instance"},
            "splitter": {"levels": ["EBSTSplitter", "TEBSTSplitter", "QOSplitter"],
            "type": "factor",
            "default": "EBSTSplitter", "core_model_parameter_type": "instance()"},
        "binary_split": {
            "levels": [0, 1],
            "type": "factor",
            "default": 0,
            "core_model_parameter_type": "bool"},
        "stop_mem_management": {
            "levels": [0, 1],
            "type": "factor",
            "default": 0,
            "core_model_parameter_type": "bool"}}}
        and
        v = {'grace_period': 200,
            'max_depth': 10,
            'delta': 1e-07,
            'tau': 0.05,
            'leaf_prediction': 0,
            'leaf_model': 0,
            'model_selector_decay': 0.95,
            'splitter': 1,
            'min_samples_split': 9,
            'binary_split': 0,
            'max_size': 500.0}
        then the function returns
        {'grace_period': 200,
        'max_depth': 10,
        'delta': 1e-07,
        'tau': 0.05,
        'leaf_prediction': 'mean',
        'leaf_model': linear_model.LinearRegression,
        'model_selector_decay': 0.95,
        'splitter': 'TEBSTSplitter',
        'min_samples_split': 9,
        'binary_split': 0,
        'max_size': 500.0}.

    Args:
        fun_control (dict): dictionary with levels and types
        v (dict): dictionary with values

    Returns:
        new_dict (dict): dictionary with levels and types

    Example:
        >>> d = {"HoeffdingTreeRegressor":{
                "leaf_prediction": {"levels": ["mean", "model", "adaptive"],
                                    "type": "factor",
                                    "default": "mean",
                                    "core_model_parameter_type": "str"}}}
            v = {"leaf_prediction": 0}
            get_dict_with_levels_and_types(d, v)
            {"leaf_prediction": "mean"}
    """
    d = fun_control["core_model_hyper_dict"]
    new_dict = {}
    for key, value in v.items():
        if key in d and d[key]["type"] == "factor":
            if d[key]["core_model_parameter_type"] == "instance":
                if "class_name" in d[key]:
                    mdl = d[key]["class_name"]
                c = d[key]["levels"][value]
                new_dict[key] = class_for_name(mdl, c)
            elif d[key]["core_model_parameter_type"] == "instance()":
                mdl = d[key]["class_name"]
                c = d[key]["levels"][value]
                k = class_for_name(mdl, c)
                new_dict[key] = k()
            else:
                new_dict[key] = d[key]["levels"][value]
        else:
            new_dict[key] = v[key]
    return new_dict


def transform_hyper_parameter_values(fun_control, hyper_parameter_values):
    """
    Transform the values of the hyperparameters according to the transform function specified in f_c
    if the hyperparameter is of type "int", or "float" or "num".
    Let f_c = {"core_model_hyper_dict":{ "leaf_prediction":
    { "levels": ["mean", "model", "adaptive"], "type": "factor", "default": "mean", "core_model_parameter_type": "str"},
    "max_depth": { "type": "int", "default": 20, "transform": "transform_power_2", "lower": 2, "upper": 20}}}
    and v = {'max_depth': 20,'leaf_prediction': 'mean'} and def transform_power_2(x): return 2**x.
    The function takes f_c and v as input and returns a dictionary with the same structure as v.
    The function transforms the values of the hyperparameters according to the transform function
    specified in f_c if the hyperparameter is of type "int", or "float" or "num".
    For example, transform_hyper_parameter_values(f_c, v) returns {'max_depth': 1048576, 'leaf_prediction': 'mean'}.
    Args:
        fun_control (dict): A dictionary containing the information about the core model and the hyperparameters.
        hyper_parameter_values (dict): A dictionary containing the values of the hyperparameters.
    Returns:
        dict: A dictionary containing the values of the hyperparameters.
    Example:
        >>> import copy
        >>> from spotPython.utils.transform import transform_hyper_parameter_values
        >>> fun_control = {"core_model_hyper_dict": {"leaf_prediction": {"levels": ["mean", "model", "adaptive"],
        "type": "factor", "default": "mean", "core_model_parameter_type": "str"},
        "max_depth": {"type": "int", "default": 20, "transform": "transform_power_2", "lower": 2, "upper": 20}}}
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
