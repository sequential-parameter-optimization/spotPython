from spotPython.utils.convert import class_for_name
import numpy as np


def modify_hyper_parameter_levels(fun_control, hyperparameter, levels):
    """

    Args:
        fun_control (dict): fun_control dictionary
        hyperparameter (str): hyperparameter name
        levels (list): list of levels

    Returns:
        fun_control (dict): updated fun_control
    Example:
        >>> fun_control = {}
            core_model  = HoeffdingTreeRegressor
            fun_control.update({"core_model": core_model})
            fun_control.update({"core_model_hyper_dict": river_hyper_dict[core_model.__name__]})
            levels = ["mean", "model"]
            fun_control = modify_hyper_parameter_levels(fun_control, "leaf_prediction", levels)
    """
    fun_control["core_model_hyper_dict"][hyperparameter].update({"levels": levels})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"lower": 0})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"upper": len(levels) - 1})
    return fun_control


def modify_hyper_parameter_bounds(fun_control, hyperparameter, bounds):
    """

    Args:
        fun_control (dict): fun_control dictionary
        hyperparameter (str): hyperparameter name
        bounds (list): list of two bound values. The first value represents the lower bound
            and the second value represents the upper bound.

    Returns:
        fun_control (dict): updated fun_control
    Example:
        >>> fun_control = {}
            core_model  = HoeffdingTreeRegressor
            fun_control.update({"core_model": core_model})
            fun_control.update({"core_model_hyper_dict": river_hyper_dict[core_model.__name__]})
            bounds = [3, 11]
            fun_control = modify_hyper_parameter_levels(fun_control, "min_samples_split", bounds)
    """
    fun_control["core_model_hyper_dict"][hyperparameter].update({"lower": bounds[0]})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"upper": bounds[1]})
    return fun_control


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


def get_default_values(fun_control):
    """Get the values from the "default" keys from the dictionary fun_control as a list.
    If the key of the value has as "type" the value "int" or "float", convert the value to the corresponding type.
    Args:
        fun_control (dict): dictionary with levels and types
    Returns:
        new_dict (dict): dictionary with default values
    Example:
        >>> d = {"core_model_hyper_dict":{
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
                "splitter": {
                    "levels": ["EBSTSplitter", "TEBSTSplitter", "QOSplitter"],
                    "type": "factor",
                    "default": "EBSTSplitter",
                    "core_model_parameter_type": "instance()"},
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
        get_default_values_from_dict(d)
        {'leaf_prediction': 'mean',
        'leaf_model': 'linear_model.LinearRegression',
        'splitter': 'EBSTSplitter',
        'binary_split': 0,
        'stop_mem_management': 0}
    """
    d = fun_control["core_model_hyper_dict"]
    new_dict = {}
    for key, value in d.items():
        if value["type"] == "int":
            new_dict[key] = int(value["default"])
        elif value["type"] == "float":
            new_dict[key] = float(value["default"])
        else:
            new_dict[key] = value["default"]
    return new_dict


def get_var_type(fun_control):
    """Get the types of the values from the dictionary fun_control as a list.
    Args:
        fun_control (dict): dictionary with levels and types
    Returns:
        (list): list with types
    Example:
        >>> d = {"core_model_hyper_dict":{
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
            "splitter": {
                "levels": ["EBSTSplitter", "TEBSTSplitter", "QOSplitter"],
                "type": "factor",
                "default": "EBSTSplitter",
                "core_model_parameter_type": "instance()"},
            "binary_split": {
                "levels": [0, 1],
                "type": "factor",
                "default": 0,
                "core_model_parameter_type": "bool"},
            "stop_mem_management": {                                                         "levels": [0, 1],
                "type": "factor",
                "default": 0,
                "core_model_parameter_type": "bool"}}}

        get_var_type(d)
        ['factor', 'factor', 'factor', 'factor', 'factor']
    """
    return list(
        fun_control["core_model_hyper_dict"][key]["type"] for key in fun_control["core_model_hyper_dict"].keys()
    )


def get_var_name(fun_control):
    """Get the names of the values from the dictionary fun_control as a list.
    Args:
        fun_control (dict): dictionary with names
    Returns:
        (list): list with names
    Example:
        >>> d = {"core_model_hyper_dict":{
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
            "splitter": {
                "levels": ["EBSTSplitter", "TEBSTSplitter", "QOSplitter"],
                "type": "factor",
                "default": "EBSTSplitter",
                "core_model_parameter_type": "instance()"},
            "binary_split": {
                "levels": [0, 1],
                "type": "factor",
                "default": 0,
                "core_model_parameter_type": "bool"},
            "stop_mem_management": {                                                         "levels": [0, 1],
                "type": "factor",
                "default": 0,
                "core_model_parameter_type": "bool"}}}

        get_var_name(d)
        ['leaf_prediction', 'leaf_model', 'splitter', 'binary_split', 'stop_mem_management']
    """
    return list(fun_control["core_model_hyper_dict"].keys())


def get_bound_values(fun_control: dict, bound: str, as_list=False) -> list or np.array:
    """Generate a list from a dictionary.
    It takes the values from the keys "bound" in the
    fun_control[]"core_model_hyper_dict"] dictionary and
    returns a list of the values in the same order as the keys in the
    dictionary.
    For example if the dictionary is
    {"a": {"upper": 1}, "b": {"upper": 2}}
    the list is [1, 2] if bound="upper".
    Args:
        fun_control (dict): dictionary with upper values
        bound (str): either "upper" or "lower"
    Returns:
        (list): list with lower or upper values
    """
    # Throw value error if bound is not upper or lower:
    if bound not in ["upper", "lower"]:
        raise ValueError("bound must be either 'upper' or 'lower'")
    d = fun_control["core_model_hyper_dict"]
    b = []
    for key, value in d.items():
        b.append(value[bound])
    if as_list:
        return b
    else:
        return np.array(b)
