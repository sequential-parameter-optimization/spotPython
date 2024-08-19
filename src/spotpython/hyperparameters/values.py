import numpy as np
import copy
import json
from sklearn.pipeline import make_pipeline

from typing import Union, List, Dict, Generator, Any
from spotpython.utils.convert import class_for_name
from spotpython.utils.transform import transform_hyper_parameter_values

# Begin Important, do not delete the following imports, they are needed for the function add_core_model_to_fun_control
import river
from river import compose
from river import forest, tree, linear_model, rules

# from river import preprocessing
import river.preprocessing

import sklearn
import sklearn.metrics

# from sklearn import ensemble, linear_model, neighbors, svm
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.svm
import sklearn.preprocessing

import spotpython
from spotpython.light import regression

# End Important


def generate_one_config_from_var_dict(
    var_dict: Dict[str, np.ndarray],
    fun_control: Dict[str, Union[List[str], str]],
    default: bool = False,
) -> Generator[Dict[str, Union[int, float]], None, None]:
    """Generate one configuration from a dictionary of variables (as a generator).

    This function takes a dictionary of variables as input arguments and returns a generator
    that yields dictionaries with the values from the arrays in the input dictionary.

    Args:
        var_dict (dict):
            A dictionary where keys are variable names and values are numpy arrays.
        fun_control (dict):
            A dictionary which (at least) has an entry with the following key:
            "var_type" (list): A list of variable types. If the entry is not "num" the corresponding
            value will be converted to the type "int".
        default (bool):
            A boolean value indicating whether to use the default values from fun_control.

    Returns:
        Generator[dict]: A generator that yields dictionaries with the values from the arrays in the input dictionary.

    Examples:
        >>> import numpy as np
        >>> from spotpython.hyperparameters.values import generate_one_config_from_var_dict
        >>> var_dict = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}
        >>> fun_control = {"var_type": ["int", "num"]}
        >>> list(generate_one_config_from_var_dict(var_dict, fun_control))
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
    """
    for values in iterate_dict_values(var_dict):
        values = convert_keys(values, fun_control["var_type"])
        values = get_dict_with_levels_and_types(fun_control=fun_control, v=values, default=default)
        values = transform_hyper_parameter_values(fun_control=fun_control, hyper_parameter_values=values)
        yield values


def return_conf_list_from_var_dict(
    var_dict: Dict[str, np.ndarray],
    fun_control: Dict[str, Union[List[str], str]],
    default: bool = False,
) -> List[Dict[str, Union[int, float]]]:
    """Return a list of configurations from a dictionary of variables.

    This function takes a dictionary of variables and a dictionary of function control as input arguments.
    It performs similar steps as generate_one_config_from_var_dict() but returns a list of dictionaries
    of hyper parameter values.

    Args:
        var_dict (dict): A dictionary where keys are variable names and values are numpy arrays.
        fun_control (dict): A dictionary which (at least) has an entry with the following key:
            "var_type" (list): A list of variable types. If the entry is not "num" the corresponding
            value will be converted to the type "int".

    Returns:
        list: A list of dictionaries of hyper parameter values. Transformations are applied to the values.

    Examples:
        >>> import numpy as np
        >>> from spotpython.hyperparameters.values import return_conf_list_from_var_dict
        >>> var_dict = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}
        >>> fun_control = {'var_type': ['int', 'int']}
        >>> return_conf_list_from_var_dict(var_dict, fun_control)
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
    """
    conf_list = []
    for values in generate_one_config_from_var_dict(var_dict, fun_control, default=default):
        conf_list.append(values)
    return conf_list


def iterate_dict_values(var_dict: Dict[str, np.ndarray]) -> Generator[Dict[str, Union[int, float]], None, None]:
    """Iterate over the values of a dictionary of variables.
    This function takes a dictionary of variables as input arguments and returns a generator that
    yields dictionaries with the values from the arrays in the input dictionary.

    Args:
        var_dict (dict): A dictionary where keys are variable names and values are numpy arrays.

    Returns:
        Generator[dict]:
            A generator that yields dictionaries with the values from the arrays in the input dictionary.

    Examples:
        >>> import numpy as np
        >>> from spotpython.hyperparameters.values import iterate_dict_values
        >>> var_dict = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}
        >>> list(iterate_dict_values(var_dict))
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
    """
    n = len(next(iter(var_dict.values())))
    for i in range(n):
        yield {key: value[i] for key, value in var_dict.items()}


def convert_keys(d: Dict[str, Union[int, float, str]], var_type: List[str]) -> Dict[str, Union[int, float]]:
    """Convert values in a dictionary to integers based on a list of variable types.
    This function takes a dictionary `d` and a list of variable types `var_type` as arguments.
    For each key in the dictionary,
    if the corresponding entry in `var_type` is not equal to `"num"`,
    the value associated with that key is converted to an integer.

    Args:
        d (dict): The input dictionary.
        var_type (list):
            A list of variable types. If the entry is not `"num"` the corresponding
            value will be converted to the type `"int"`.

    Returns:
        dict: The modified dictionary with values converted to integers based on `var_type`.

    Examples:
        >>> from spotpython.hyperparameters.values import convert_keys
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


def get_dict_with_levels_and_types(fun_control: Dict[str, Any], v: Dict[str, Any], default=False) -> Dict[str, Any]:
    """Get dictionary with levels and types.
    The function maps the numerical output of the hyperparameter optimization to the corresponding levels
    of the hyperparameter needed by the core model, i.e., the tuned algorithm.
    The function takes the dictionaries fun_control and v and returns a new dictionary with the same keys as v
    but with the values of the levels of the keys from fun_control.
    If the key value in the dictionary is 0, it takes the first value from the list,
    if it is 1, it takes the second and so on.
    If a key is not in fun_control, it takes the key from v.
    If the core_model_parameter_type value is instance, it returns the class of the value from the module
    via getattr("class", value).

    Args:
        fun_control (Dict[str, Any]):
            A dictionary containing information about the core model hyperparameters.
        v (Dict[str, Any]):
            A dictionary containing the numerical output of the hyperparameter optimization.
        default (bool):
            A boolean value indicating whether to use the default values from fun_control.

    Returns:
        Dict[str, Any]:
            A new dictionary with the same keys as v but with the values of the levels of the keys from fun_control.

    Examples:
        >>> fun_control = {
        ...     "core_model_hyper_dict": {
        ...         "leaf_prediction": {
        ...             "levels": ["mean", "model", "adaptive"],
        ...             "type": "factor",
        ...             "default": "mean",
        ...             "core_model_parameter_type": "str"
        ...         },
        ...         "leaf_model": {
        ...             "levels": [
        ...                 "linear_model.LinearRegression",
        ...                 "linear_model.PARegressor",
        ...                 "linear_model.Perceptron"
        ...             ],
        ...             "type": "factor",
        ...             "default": "LinearRegression",
        ...             "core_model_parameter_type": "instance"
        ...         },
        ...         "splitter": {
        ...             "levels": ["EBSTSplitter", "TEBSTSplitter", "QOSplitter"],
        ...             "type": "factor",
        ...             "default": "EBSTSplitter",
        ...             "core_model_parameter_type": "instance()"
        ...         },
        ...         "binary_split": {
        ...             "levels": [0, 1],
        ...             "type": "factor",
        ...             "default": 0,
        ...             "core_model_parameter_type": "bool"
        ...         },
        ...         "stop_mem_management": {
        ...             "levels": [0, 1],
        ...             "type": "factor",
        ...             "default": 0,
        ...             "core_model_parameter_type": "bool"
        ...         }
        ...     }
        ... }
        >>> v = {
        ...     'grace_period': 200,
        ...     'max_depth': 10,
        ...     'delta': 1e-07,
        ...     'tau': 0.05,
        ...     'leaf_prediction': 0,
        ...     'leaf_model': 0,
        ...     'model_selector_decay': 0.95,
        ...     'splitter': 1,
        ...     'min_samples_split': 9,
        ...     'binary_split': 0,
        ...     'max_size': 500.0
        ... }
        >>> get_dict_with_levels_and_types(fun_control, v)
        {
            'grace_period': 200,
            'max_depth': 10,
            'delta': 1e-07,
            'tau': 0.05,
            'leaf_prediction': 'mean',
            'leaf_model': linear_model.LinearRegression,
            'model_selector_decay': 0.95,
            'splitter': TEBSTSplitter,
            'min_samples_split': 9,
            'binary_split': False,
            'max_size': 500.0
        }
    """
    if default:
        d = fun_control["core_model_hyper_dict_default"]
    else:
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
            # bool() introduced to convert 0 and 1 to False and True in v0.14.54
            elif d[key]["core_model_parameter_type"] == "bool":
                new_dict[key] = bool(d[key]["levels"][value])
            else:
                new_dict[key] = d[key]["levels"][value]
        else:
            new_dict[key] = v[key]
    return new_dict


def assign_values(X: np.array, var_list: list) -> dict:
    """
    This function takes an np.array X and a list of variable names as input arguments
    and returns a dictionary where the keys are the variable names and the values are assigned from X.

    Args:
        X (np.array):
            A 2D numpy array where each column represents a variable.
        var_list (list):
            A list of strings representing variable names.

    Returns:
        dict:
            A dictionary where keys are variable names and values are assigned from X.

    Examples:
        >>> import numpy as np
        >>> from spotpython.hyperparameters.values import assign_values
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> var_list = ['a', 'b']
        >>> result = assign_values(X, var_list)
        >>> print(result)
        {'a': array([1, 3, 5]), 'b': array([2, 4, 6])}
    """
    result = {}
    for i, var_name in enumerate(var_list):
        result[var_name] = X[:, i]
    return result


def modify_boolean_hyper_parameter_levels(fun_control, hyperparameter, levels) -> None:
    """
    This function modifies the levels of a boolean hyperparameter in the fun_control dictionary.
    It also sets the lower and upper bounds of the hyperparameter to 0 and len(levels) - 1, respectively.

    Args:
        fun_control (dict):
            fun_control dictionary
        hyperparameter (str):
            hyperparameter name
        levels (list):
            list of levels

    Returns:
        None.
    """
    fun_control["core_model_hyper_dict"][hyperparameter].update({"levels": levels})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"lower": levels[0]})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"upper": levels[1]})


def modify_hyper_parameter_levels(fun_control, hyperparameter, levels) -> None:
    """
    This function modifies the levels of a hyperparameter in the fun_control dictionary.
    It also sets the lower and upper bounds of the hyperparameter to 0 and len(levels) - 1, respectively.

    Args:
        fun_control (dict):
            fun_control dictionary
        hyperparameter (str):
            hyperparameter name
        levels (list):
            list of levels

    Returns:
        None.

    Examples:
        >>> fun_control = {}
            from spotpython.hyperparameters.values import modify_hyper_parameter_levels
            core_model  = HoeffdingTreeRegressor
            fun_control.update({"core_model": core_model})
            fun_control.update({"core_model_hyper_dict": river_hyper_dict[core_model.__name__]})
            levels = ["mean", "model"]
            fun_control = modify_hyper_parameter_levels(fun_control, "leaf_prediction", levels)
    """
    fun_control["core_model_hyper_dict"][hyperparameter].update({"levels": levels})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"lower": 0})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"upper": len(levels) - 1})


def modify_hyper_parameter_bounds(fun_control, hyperparameter, bounds) -> None:
    """
    Modify the bounds of a hyperparameter in the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary
        hyperparameter (str):
            hyperparameter name
        bounds (list):
            list of two bound values. The first value represents the lower bound
            and the second value represents the upper bound.

    Returns:
        None.

    Examples:
        >>> from spotpython.hyperparameters.values import modify_hyper_parameter_levels
            fun_control = {}
            core_model  = HoeffdingTreeRegressor
            fun_control.update({"core_model": core_model})
            fun_control.update({"core_model_hyper_dict": river_hyper_dict[core_model.__name__]})
            bounds = [3, 11]
            fun_control = modify_hyper_parameter_levels(fun_control, "min_samples_split", bounds)
    """
    fun_control["core_model_hyper_dict"][hyperparameter].update({"lower": bounds[0]})
    fun_control["core_model_hyper_dict"][hyperparameter].update({"upper": bounds[1]})


def get_default_values(fun_control) -> dict:
    """Get the values from the "default" keys from the dictionary fun_control as a dict.
    If the key of the value has as "type" the value "int" or "float", convert the value to the corresponding type.

    Args:
        fun_control (dict):
            dictionary with levels and types

    Returns:
        new_dict (dict):
            dictionary with default values

    Examples:
        >>> from spotpython.hyperparameters.values import get_default_values
            d = {"core_model_hyper_dict":{
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
            get_default_values(d)
            {'leaf_prediction': 'mean',
            'leaf_model': 'linear_model.LinearRegression',
            'splitter': 'EBSTSplitter',
            'binary_split': 0,
            'stop_mem_management': 0}
    """
    d = fun_control["core_model_hyper_dict_default"]
    new_dict = {}
    for key, value in d.items():
        if value["type"] == "int":
            new_dict[key] = int(value["default"])
        elif value["type"] == "float":
            new_dict[key] = float(value["default"])
        else:
            new_dict[key] = value["default"]
    return new_dict


def get_var_type(fun_control) -> list:
    """
    Get the types of the values from the dictionary fun_control as a list.
    If no "core_model_hyper_dict" key exists in fun_control, return None.

    Args:
        fun_control (dict):
            dictionary with levels and types

    Returns:
        (list):
            list with types

    Examples:
        >>> from spotpython.hyperparameters.values import get_var_type
            d = {"core_model_hyper_dict":{
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
    if "core_model_hyper_dict" not in fun_control.keys():
        return None
    else:
        return list(
            fun_control["core_model_hyper_dict"][key]["type"] for key in fun_control["core_model_hyper_dict"].keys()
        )


def get_transform(fun_control) -> list:
    """Get the transformations of the values from the dictionary fun_control as a list.

    Args:
        fun_control (dict):
            dictionary with levels and types

    Returns:
        (list):
            list with transformations

    Examples:
        >>> from spotpython.hyperparameters.values import get_transform
            d = {"core_model_hyper_dict":{
            "leaf_prediction": {
                "levels": ["mean", "model", "adaptive"],
                "type": "factor",
                "default": "mean",
                "transform": "None",
                "core_model_parameter_type": "str"},
            "leaf_model": {
                "levels": ["linear_model.LinearRegression", "linear_model.PARegressor", "linear_model.Perceptron"],
                "type": "factor",
                "default": "LinearRegression",
                "transform": "None",
                "core_model_parameter_type": "instance"},
            "splitter": {
                "levels": ["EBSTSplitter", "TEBSTSplitter", "QOSplitter"],
                "type": "factor",
                "default": "EBSTSplitter",
                "transform": "None",
                "core_model_parameter_type": "instance()"},
            "binary_split": {
                "levels": [0, 1],
                "type": "factor",
                "default": 0,
                "transform": "None",
                "core_model_parameter_type": "bool"},
            "stop_mem_management": {                                                         "levels": [0, 1],
                "type": "factor",
                "default": 0,
                "transform": "None",
                "core_model_parameter_type": "bool"}}}

            get_transform(d)
            ['None', 'None', 'None', 'None', 'None']
    """
    return list(
        fun_control["core_model_hyper_dict"][key]["transform"] for key in fun_control["core_model_hyper_dict"].keys()
    )


def get_var_name(fun_control) -> list:
    """Get the names of the values from the dictionary fun_control as a list.
    If no "core_model_hyper_dict" key exists in fun_control, return None.

    Args:
        fun_control (dict):
            dictionary with names

    Returns:
        (list):
            ist with names

    Examples:
        >>> from spotpython.hyperparameters.values import get_var_name
            fun_control = {"core_model_hyper_dict":{
                        "leaf_prediction": {
                            "levels": ["mean", "model", "adaptive"],
                            "type": "factor",
                            "default": "mean",
                            "core_model_parameter_type": "str"},
                        "leaf_model": {
                            "levels": ["linear_model.LinearRegression",
                                        "linear_model.PARegressor",
                                        "linear_model.Perceptron"],
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
            get_var_name(fun_control)
            ['leaf_prediction',
                'leaf_model',
                'splitter',
                'binary_split',
                'stop_mem_management']
    """
    if "core_model_hyper_dict" not in fun_control.keys():
        return None
    else:
        return list(fun_control["core_model_hyper_dict"].keys())


def get_bound_values(fun_control: dict, bound: str, as_list: bool = False) -> Union[List, np.ndarray]:
    """Generate a list or array from a dictionary.
    This function takes the values from the keys "bound" in the
    fun_control["core_model_hyper_dict"] dictionary and returns a list or array of the values
    in the same order as the keys in the dictionary.

    Args:
        fun_control (dict):
            A dictionary containing a key "core_model_hyper_dict"
            which is a dictionary with keys that have either an "upper" or "lower" value.
        bound (str):
            Either "upper" or "lower",
            indicating which value to extract from the inner dictionary.
        as_list (bool):
            If True, return a list.
            If False, return a numpy array. Default is False.

    Returns:
        list or np.ndarray:
            A list or array of the extracted values.

    Raises:
        ValueError:
            If bound is not "upper" or "lower".

    Examples:
        >>> from spotpython.hyperparameters.values import get_bound_values
        >>> fun_control = {"core_model_hyper_dict": {"a": {"upper": 1}, "b": {"upper": 2}}}
        >>> get_bound_values(fun_control, "upper", as_list=True)
        [1, 2]
    """
    # Throw value error if bound is not upper or lower:
    if bound not in ["upper", "lower"]:
        raise ValueError("bound must be either 'upper' or 'lower'")
    # check if key "core_model_hyper_dict" exists in fun_control:
    if "core_model_hyper_dict" not in fun_control.keys():
        return None
    else:
        d = fun_control["core_model_hyper_dict"]
        b = []
        for key, value in d.items():
            b.append(value[bound])
        if as_list:
            return b
        else:
            return np.array(b)


def replace_levels_with_positions(hyper_dict, hyper_dict_values) -> dict:
    """Replace the levels with the position in the levels list.
    The function that takes two dictionaries.
    The first contains as hyperparameters as keys.
    If the hyperparameter has the key "levels",
    then the value of the corresponding hyperparameter in the second dictionary is
    replaced by the position of the value in the list of levels.
    The function returns a dictionary with the same keys as the second dictionary.
    For example, if the second dictionary is {"a": 1, "b": "model1", "c": 3}
    and the first dictionary is {
        "a": {"type": "int"},
        "b": {"levels": ["model4", "model5", "model1"]},
        "d": {"type": "float"}},
    then the function should return {"a": 1, "b": 2, "c": 3}.

    Args:
        hyper_dict (dict):
            dictionary with levels
        hyper_dict_values (dict):
            dictionary with values

    Returns:
        (dict):
            dictionary with values

    Examples:
        >>> from spotpython.hyperparameters.values import replace_levels_with_positions
            hyper_dict = {"leaf_prediction": {
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
                    "core_model_parameter_type": "bool"}}
            hyper_dict_values = {"leaf_prediction": "mean",
                "leaf_model": "linear_model.LinearRegression",
                "splitter": "EBSTSplitter",
                "binary_split": 0,
                "stop_mem_management": 0}
            replace_levels_with_position(hyper_dict, hyper_dict_values)
                {'leaf_prediction': 0,
                'leaf_model': 0,
                'splitter': 0,
                'binary_split': 0,
                'stop_mem_management': 0}
    """
    hyper_dict_values_new = copy.deepcopy(hyper_dict_values)
    # generate an error if the following code fails and write an error message:
    try:
        for key, value in hyper_dict_values.items():
            if key in hyper_dict.keys():
                if "levels" in hyper_dict[key].keys():
                    hyper_dict_values_new[key] = hyper_dict[key]["levels"].index(value)
    except Exception as e:
        print("!!! Warning: ", e)
        print("Did you modify lower and upper bounds so that the default values are not included?")
        print("Returning 'None'.")
        return None
    return hyper_dict_values_new


def get_values_from_dict(dictionary) -> np.array:
    """Get the values from a dictionary as an array.
    Generate an np.array that contains the values of the keys of a dictionary
    in the same order as the keys of the dictionary.

    Args:
        dictionary (dict):
            dictionary with values

    Returns:
        (np.array):
            array with values

    Examples:
        >>> from spotpython.hyperparameters.values import get_values_from_dict
        >>> d = {"a": 1, "b": 2, "c": 3}
        >>> get_values_from_dict(d)
        array([1, 2, 3])
    """
    return np.array(list(dictionary.values()))


def add_core_model_to_fun_control(fun_control, core_model, hyper_dict=None, filename=None) -> dict:
    """Add the core model to the function control dictionary. It updates the keys "core_model",
    "core_model_hyper_dict", "var_type", "var_name" in the fun_control dictionary.

    Args:
        fun_control (dict):
            The fun_control dictionary.
        core_model (class):
            The core model.
        hyper_dict (dict):
            The hyper parameter dictionary. Optional. Default is None. If no hyper_dict is provided,
            the function will try to load the hyper_dict from the file specified by filename.
        filename (str):
            The name of the json file that contains the hyper parameter dictionary.
            Optional. Default is None. If no filename is provided, the function will try to load the
            hyper_dict from the hyper_dict dictionary.

    Returns:
        (dict):
            The updated fun_control dictionary.

    Notes:
        The function adds the following keys to the fun_control dictionary:
        "core_model": The core model.
        "core_model_hyper_dict": The hyper parameter dictionary for the core model.
        "core_model_hyper_dict_default": The hyper parameter dictionary for the core model.
        "var_type": A list of variable types.
        "var_name": A list of variable names.
        The original hyperparameters of the core model are stored in the "core_model_hyper_dict_default" key.
        These remain unmodified, while the "core_model_hyper_dict" key is modified during the tuning process.

    Examples:
        >>> from spotpython.light.regression.netlightregression import NetLightRegression
            from spotpython.hyperdict.light_hyper_dict import LightHyperDict
            from spotpython.hyperparameters.values import add_core_model_to_fun_control
            add_core_model_to_fun_control(fun_control=fun_control,
                                        core_model=NetLightRegression,
                                        hyper_dict=LightHyperDict)
            # or, if a user wants to use a custom hyper_dict:
        >>> from spotpython.light.regression.netlightregression import NetLightRegression
            from spotpython.hyperparameters.values import add_core_model_to_fun_control
            add_core_model_to_fun_control(fun_control=fun_control,
                                        core_model=NetLightRegression,
                                        filename="./hyperdict/user_hyper_dict.json")

    """
    fun_control.update({"core_model": core_model})
    if filename is None:
        new_hyper_dict = hyper_dict().load()
    else:
        with open(filename, "r") as f:
            new_hyper_dict = json.load(f)
    fun_control.update({"core_model_hyper_dict": new_hyper_dict[core_model.__name__]})
    fun_control.update({"core_model_hyper_dict_default": copy.deepcopy(new_hyper_dict[core_model.__name__])})
    var_type = get_var_type(fun_control)
    var_name = get_var_name(fun_control)
    lower = get_bound_values(fun_control, "lower", as_list=False)
    upper = get_bound_values(fun_control, "upper", as_list=False)
    fun_control.update({"var_type": var_type, "var_name": var_name, "lower": lower, "upper": upper})


def get_one_core_model_from_X(
    X,
    fun_control=None,
    default=False,
):
    """Get one core model from X.

    Args:
        X (np.array):
            The array with the hyper parameter values.
        fun_control (dict):
            The function control dictionary.
        default (bool):
            A boolean value indicating whether to use the default values from fun_control.

    Returns:
        (class):
            The core model.

    Examples:
        >>> from river.tree import HoeffdingAdaptiveTreeRegressor
            from spotriver.data.river_hyper_dict import RiverHyperDict
            fun_control = {}
            add_core_model_to_fun_control(core_model=HoeffdingAdaptiveTreeRegressor,
                fun_control=fun_control,
                hyper_dict=RiverHyperDict,
                filename=None)
            X = np.array([0, 0, 0, 0, 0])
            get_one_core_model_from_X(X, fun_control)
            HoeffdingAdaptiveTreeRegressor()
    """
    var_dict = assign_values(X, fun_control["var_name"])
    # var_dict = assign_values(X, get_var_name(fun_control))
    config = return_conf_list_from_var_dict(var_dict, fun_control, default=default)[0]
    core_model = fun_control["core_model"](**config)
    return core_model


def get_one_config_from_X(X, fun_control=None):
    """Get one config from X.

    Args:
        X (np.array):
            The array with the hyper parameter values.
        fun_control (dict):
            The function control dictionary.

    Returns:
        (dict):
            The config dictionary.

    Examples:
        >>> from river.tree import HoeffdingAdaptiveTreeRegressor
            from spotriver.data.river_hyper_dict import RiverHyperDict
            fun_control = {}
            add_core_model_to_fun_control(core_model=HoeffdingAdaptiveTreeRegressor,
                fun_control=func_control,
                hyper_dict=RiverHyperDict,
                filename=None)
            X = np.array([0, 0, 0, 0, 0])
            get_one_config_from_X(X, fun_control)
            {'leaf_prediction': 'mean',
            'leaf_model': 'NBAdaptive',
            'splitter': 'HoeffdingAdaptiveTreeSplitter',
            'binary_split': 'info_gain',
            'stop_mem_management': False}
    """
    var_dict = assign_values(X, fun_control["var_name"])
    config = return_conf_list_from_var_dict(var_dict, fun_control)[0]
    return config


def get_one_sklearn_model_from_X(X, fun_control=None):
    """Get one sklearn model from X.

    Args:
        X (np.array):
            The array with the hyper parameter values.
        fun_control (dict):
            The function control dictionary.

    Returns:
        (class):
            The sklearn model.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
            from spotriver.data.sklearn_hyper_dict import SklearnHyperDict
            fun_control = {}
            add_core_model_to_fun_control(core_model=LinearRegression,
                fun_control=func_control,
                hyper_dict=SklearnHyperDict,
                filename=None)
            X = np.array([0, 0, 0, 0, 0])
            get_one_sklearn_model_from_X(X, fun_control)
            LinearRegression()
    """
    core_model = get_one_core_model_from_X(X=X, fun_control=fun_control)
    if fun_control["prep_model"] is not None:
        model = make_pipeline(fun_control["prep_model"], core_model)
    else:
        model = core_model
    return model


def get_one_river_model_from_X(X, fun_control=None):
    """Get one river model from X.

    Args:
        X (np.array):
            The array with the hyper parameter values.
        fun_control (dict):
            The function control dictionary.

    Returns:
        (class):
            The river model.

    Examples:
        >>> from river.tree import HoeffdingAdaptiveTreeRegressor
            from spotriver.data.river_hyper_dict import RiverHyperDict
            fun_control = {}
            add_core_model_to_fun_control(core_model=HoeffdingAdaptiveTreeRegressor,
                fun_control=func_control,
                hyper_dict=RiverHyperDict,
                filename=None)
            X = np.array([0, 0, 0, 0, 0])
            get_one_river_model_from_X(X, fun_control)
            HoeffdingAdaptiveTreeRegressor()
    """
    core_model = get_one_core_model_from_X(X=X, fun_control=fun_control)
    if fun_control["prep_model"] is not None:
        model = compose.Pipeline(fun_control["prep_model"], core_model)
    else:
        model = core_model
    return model


def get_default_hyperparameters_as_array(fun_control) -> np.array:
    """Get the default hyper parameters as array.

    Args:
        fun_control (dict):
            The function control dictionary.

    Returns:
        (np.array):
            The default hyper parameters as array.

    Examples:
        >>> from river.tree import HoeffdingAdaptiveTreeRegressor
            from spotriver.data.river_hyper_dict import RiverHyperDict
            from spotpython.hyperparameters.values import (
                get_default_hyperparameters_as_array,
                add_core_model_to_fun_control)
            fun_control = {}
            add_core_model_to_fun_control(core_model=HoeffdingAdaptiveTreeRegressor,
                fun_control=func_control,
                hyper_dict=RiverHyperDict,
                filename=None)
            get_default_hyperparameters_as_array(fun_control)
            array([0, 0, 0, 0, 0])
    """
    X0 = get_default_values(fun_control)
    X0 = replace_levels_with_positions(fun_control["core_model_hyper_dict_default"], X0)
    if X0 is None:
        return None
    else:
        X0 = get_values_from_dict(X0)
        X0 = np.array([X0])
        X0.shape[1]
        return X0


# def get_default_hyperparameters_for_core_model(fun_control) -> dict:
#     """Get the default hyper parameters for the core model.

#     Args:
#         fun_control (dict):
#             The function control dictionary.

#     Returns:
#         (dict):
#             The default hyper parameters for the core model.

#     Examples:
#         >>> from river.tree import HoeffdingAdaptiveTreeRegressor
#             from spotriver.data.river_hyper_dict import RiverHyperDict
#             fun_control = {}
#             add_core_model_to_fun_control(core_model=HoeffdingAdaptiveTreeRegressor,
#                 fun_control=func_control,
#                 hyper_dict=RiverHyperDict,
#                 filename=None)
#             get_default_hyperparameters_for_core_model(fun_control)
#             {'leaf_prediction': 'mean',
#             'leaf_model': 'NBAdaptive',
#             'splitter': 'HoeffdingAdaptiveTreeSplitter',
#             'binary_split': 'info_gain',
#             'stop_mem_management': False}
#     """
#     values = get_default_values(fun_control)
#     print(f"values: {values}")
#     pprint.pprint(fun_control)
#     values = get_dict_with_levels_and_types(fun_control=fun_control, v=values, default=True)
#     values = convert_keys(values, fun_control["var_type"])
#     values = transform_hyper_parameter_values(fun_control=fun_control, hyper_parameter_values=values)
#     return values


def get_tuned_architecture(spot_tuner, fun_control, force_minX=False) -> dict:
    """
    Returns the tuned architecture. If the spot tuner has noise,
    it returns the architecture with the lowest mean (.min_mean_X),
    otherwise it returns the architecture with the lowest value (.min_X).

    Args:
        spot_tuner (object):
            spot tuner object.
        fun_control (dict):
            dictionary containing control parameters for the hyperparameter tuning.
        force_minX (bool):
            If True, return the architecture with the lowest value (.min_X).

    Returns:
        (dict):
            dictionary containing the tuned architecture.
    """
    if not spot_tuner.noise or force_minX:
        X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1, -1))
    else:
        # noise or force_minX is False:
        X = spot_tuner.to_all_dim(spot_tuner.min_mean_X.reshape(1, -1))
    config = get_one_config_from_X(X, fun_control)
    return config


def create_model(config, fun_control, **kwargs) -> object:
    """
    Creates a model for the given configuration and control parameters.

    Args:
        config (dict):
            dictionary containing the configuration for the hyperparameter tuning.
        fun_control (dict):
            dictionary containing control parameters for the hyperparameter tuning.
        **kwargs (Any):
            additional keyword arguments.

    Returns:
        (object):
            model object.
    """
    return fun_control["core_model"](**config, **kwargs)


def set_control_key_value(control_dict, key, value, replace=False) -> None:
    """
    This function sets the key value pair in the control_dict dictionary.

    Args:
        control_dict (dict):
            control_dict dictionary
        key (str): key
        value (Any): value
        replace (bool): replace value if key already exists. Default is False.

    Returns:
        None.

    Attributes:
        key (str): key
        value (Any): value

    Examples:
        >>> from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import set_control_key_value
            control_dict = fun_control_init()
            set_control_key_value(control_dict=control_dict,
                          key="key",
                          value="value")
            control_dict["key"]

    """
    if replace:
        control_dict.update({key: value})
    else:
        if key not in control_dict.keys():
            control_dict.update({key: value})


def set_control_hyperparameter_value(control_dict, hyperparameter, value) -> None:
    """
    This function sets the hyperparameter values depending on the var_type
    via modify_hyperameter_levels or modify_hyperparameter_bounds in the control_dict dictionary.
    If the hyperparameter is a factor, it calls modify_hyper_parameter_levels.
    Otherwise, it calls modify_hyper_parameter_bounds.

    Args:
        control_dict (dict):
            control_dict dictionary
        hyperparameter (str): key
        value (Any): value

    Returns:
        None.

    """
    print(f"Setting hyperparameter {hyperparameter} to value {value}.")
    vt = get_var_type_from_var_name(fun_control=control_dict, var_name=hyperparameter)
    print(f"Variable type is {vt}.")
    core_type = get_core_model_parameter_type_from_var_name(fun_control=control_dict, var_name=hyperparameter)
    print(f"Core type is {core_type}.")
    if vt == "factor" and core_type != "bool":
        print("Calling modify_hyper_parameter_levels().")
        modify_hyper_parameter_levels(fun_control=control_dict, hyperparameter=hyperparameter, levels=value)
    elif vt == "factor" and core_type == "bool":
        print("Calling modify_boolean_hyper_parameter_levels().")
        modify_boolean_hyper_parameter_levels(fun_control=control_dict, hyperparameter=hyperparameter, levels=value)
    else:
        print("Calling modify_hyper_parameter_bounds().")
        modify_hyper_parameter_bounds(fun_control=control_dict, hyperparameter=hyperparameter, bounds=value)


def get_control_key_value(control_dict=None, key=None) -> Any:
    """
    This function gets the key value pair from the control_dict dictionary.
    If the key does not exist, return None.
    If the control_dict dictionary is None, return None.

    Args:
        control_dict (dict):
            control_dict dictionary
        key (str): key

    Returns:
        value (Any):
            value

    Examples:
        >>> from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import get_control_key_value
            control_dict = fun_control_init()
            get_control_key_value(control_dict=control_dict,
                            key="key")
            "value"
    """
    if control_dict is None:
        return None
    else:
        # check if key "core_model_hyper_dict" exists in fun_control:
        if "core_model_hyper_dict" in control_dict.keys():
            if key == "lower":
                lower = get_bound_values(fun_control=control_dict, bound="lower")
                return lower
            if key == "upper":
                upper = get_bound_values(fun_control=control_dict, bound="upper")
                return upper
            if key == "var_name":
                var_name = get_var_name(fun_control=control_dict)
                return var_name
            if key == "var_type":
                var_type = get_var_type(fun_control=control_dict)
                return var_type
            if key == "transform":
                transform = get_transform(fun_control=control_dict)
                return transform
        # check if key exists in control_dict:
        elif control_dict is None or key not in control_dict.keys():
            return None
        else:
            return control_dict[key]


def get_var_type_from_var_name(fun_control, var_name) -> str:
    """
    This function gets the variable type from the variable name.

    Args:
        fun_control (dict): fun_control dictionary
        var_name (str): variable name

    Returns:
        (str): variable type

    Examples:
        >>> from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import get_var_type_from_var_name
            control_dict = fun_control_init()
            get_var_type_from_var_name(var_name="max_depth",
                            fun_control=control_dict)
            "int"
    """
    var_type_list = get_control_key_value(control_dict=fun_control, key="var_type")
    var_name_list = get_control_key_value(control_dict=fun_control, key="var_name")
    return var_type_list[var_name_list.index(var_name)]


def get_core_model_parameter_type_from_var_name(fun_control, var_name) -> str:
    """
    Extracts the core_model_parameter_type value from a dictionary for a specified key.

    Args:
        fun_control (dict):
            The dictionary containing the information.
        var_name (str):
            The key for which to extract the core_model_parameter_type value.

    Returns:
        (str):
            The core_model_parameter_type value if available, else None.
    """
    # Check if the key exists in the dictionary and it has a 'core_model_parameter_type' entry
    if (
        var_name in fun_control["core_model_hyper_dict"]
        and "core_model_parameter_type" in fun_control["core_model_hyper_dict"][var_name]
    ):
        return fun_control["core_model_hyper_dict"][var_name]["core_model_parameter_type"]
    else:
        return None


def get_ith_hyperparameter_name_from_fun_control(fun_control, key, i):
    """
    Get the ith hyperparameter name from the fun_control dictionary.

    Args:
        fun_control (dict): fun_control dictionary
        key (str): key
        i (int): index

    Returns:
        (str): hyperparameter name

    Examples:
        >>> from spotpython.utils.device import getDevice
            from spotpython.utils.init import fun_control_init
            from spotpython.utils.file import get_experiment_name
            import numpy as np
            from spotpython.data.diabetes import Diabetes
            from spotpython.light.regression.netlightregression import NetLightRegression
            from spotpython.hyperdict.light_hyper_dict import LightHyperDict
            from spotpython.hyperparameters.values import add_core_model_to_fun_control
            from spotpython.hyperparameters.values import get_ith_hyperparameter_name_from_fun_control
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.hyperparameters.values import set_control_hyperparameter_value
            experiment_name = get_experiment_name(prefix="000")
            fun_control = fun_control_init(
                _L_in=10,
                _L_out=1,
                TENSORBOARD_CLEAN=True,
                device=getDevice(),
                enable_progress_bar=False,
                fun_evals=15,
                log_level=10,
                max_time=1,
                num_workers=0,
                show_progress=True,
                tolerance_x=np.sqrt(np.spacing(1)),
                )
            dataset = Diabetes()
            set_control_key_value(control_dict=fun_control,
                                    key="data_set",
                                    value=dataset,
                                    replace=True)
            add_core_model_to_fun_control(core_model=NetLightRegression,
                                        fun_control=fun_control,
                                        hyper_dict=LightHyperDict)

            set_control_hyperparameter_value(fun_control, "l1", [3,8])
            set_control_hyperparameter_value(fun_control, "optimizer", ["Adam", "AdamW", "Adamax", "NAdam"])
            get_ith_hyperparameter_name_from_fun_control(fun_control, key="optimizer", i=0)
            Adam

    """
    if "core_model_hyper_dict" in fun_control:
        if key in fun_control["core_model_hyper_dict"]:
            if "levels" in fun_control["core_model_hyper_dict"][key]:
                if i < len(fun_control["core_model_hyper_dict"][key]["levels"]):
                    return fun_control["core_model_hyper_dict"][key]["levels"][i]
    return None


def get_tuned_hyperparameters(spot_tuner, fun_control=None) -> dict:
    """
    Get the tuned hyperparameters from the spot tuner.
    This is just a wrapper function for the spot `get_tuned_hyperparameters` method.

    Args:
        spot_tuner (object):
            spot tuner object.
        fun_control (dict):
            dictionary containing control parameters for the hyperparameter tuning.
            Optional. Default is None.

    Returns:
        (dict):
            dictionary containing the tuned hyperparameters.

    Examples:
        >>> from spotpython.utils.device import getDevice
            from math import inf
            from spotpython.utils.init import fun_control_init
            import numpy as np
            from spotpython.hyperparameters.values import set_control_key_value
            from spotpython.data.diabetes import Diabetes
            from spotpython.hyperparameters.values import get_tuned_hyperparameters
            MAX_TIME = 1
            FUN_EVALS = 10
            INIT_SIZE = 5
            WORKERS = 0
            PREFIX="037"
            DEVICE = getDevice()
            DEVICES = 1
            TEST_SIZE = 0.4
            TORCH_METRIC = "mean_squared_error"
            dataset = Diabetes()
            fun_control = fun_control_init(
                _L_in=10,
                _L_out=1,
                _torchmetric=TORCH_METRIC,
                PREFIX=PREFIX,
                TENSORBOARD_CLEAN=True,
                data_set=dataset,
                device=DEVICE,
                enable_progress_bar=False,
                fun_evals=FUN_EVALS,
                log_level=50,
                max_time=MAX_TIME,
                num_workers=WORKERS,
                show_progress=True,
                test_size=TEST_SIZE,
                tolerance_x=np.sqrt(np.spacing(1)),
                )
            from spotpython.light.regression.netlightregression import NetLightRegression
            from spotpython.hyperdict.light_hyper_dict import LightHyperDict
            from spotpython.hyperparameters.values import add_core_model_to_fun_control
            add_core_model_to_fun_control(fun_control=fun_control,
                                        core_model=NetLightRegression,
                                        hyper_dict=LightHyperDict)
            from spotpython.hyperparameters.values import set_control_hyperparameter_value
            set_control_hyperparameter_value(fun_control, "l1", [7, 8])
            set_control_hyperparameter_value(fun_control, "epochs", [3, 5])
            set_control_hyperparameter_value(fun_control, "batch_size", [4, 5])
            set_control_hyperparameter_value(fun_control, "optimizer", [
                            "Adam",
                            "RAdam",
                        ])
            set_control_hyperparameter_value(fun_control, "dropout_prob", [0.01, 0.1])
            set_control_hyperparameter_value(fun_control, "lr_mult", [0.5, 5.0])
            set_control_hyperparameter_value(fun_control, "patience", [2, 3])
            set_control_hyperparameter_value(fun_control, "act_fn",[
                            "ReLU",
                            "LeakyReLU"
                        ] )
            from spotpython.utils.init import design_control_init, surrogate_control_init
            design_control = design_control_init(init_size=INIT_SIZE)
            surrogate_control = surrogate_control_init(noise=True,
                                                        n_theta=2)
            from spotpython.fun.hyperlight import HyperLight
            fun = HyperLight(log_level=50).fun
            from spotpython.spot import spot
            spot_tuner = spot.Spot(fun=fun,
                                fun_control=fun_control,
                                design_control=design_control,
                                surrogate_control=surrogate_control)
            spot_tuner.run()
            get_tuned_hyperparameters(spot_tuner)
                {'l1': 7.0,
                'epochs': 5.0,
                'batch_size': 4.0,
                'act_fn': 0.0,
                'optimizer': 0.0,
                'dropout_prob': 0.01,
                'lr_mult': 5.0,
                'patience': 3.0,
                'initialization': 1.0}
    """
    return spot_tuner.get_tuned_hyperparameters(fun_control=fun_control)


def update_fun_control(fun_control, new_control) -> dict:
    for i, (key, value) in enumerate(new_control.items()):
        if new_control[key]["type"] == "int":
            set_control_hyperparameter_value(
                fun_control,
                key,
                [
                    int(new_control[key]["lower"]),
                    int(new_control[key]["upper"]),
                ],
            )
        if (new_control[key]["type"] == "factor") and (new_control[key]["core_model_parameter_type"] == "bool"):
            set_control_hyperparameter_value(
                fun_control,
                key,
                [
                    int(new_control[key]["lower"]),
                    int(new_control[key]["upper"]),
                ],
            )
        if new_control[key]["type"] == "float":
            set_control_hyperparameter_value(
                fun_control,
                key,
                [
                    float(new_control[key]["lower"]),
                    float(new_control[key]["upper"]),
                ],
            )
        if new_control[key]["type"] == "factor" and new_control[key]["core_model_parameter_type"] != "bool":
            fle = new_control[key]["levels"]
            # convert the string to a list of strings
            fle = fle.split()
            set_control_hyperparameter_value(fun_control, key, fle)
            fun_control["core_model_hyper_new_control"][key].update({"upper": len(fle) - 1})


def update_fun_control_with_hyper_num_cat_dicts(fun_control, num_dict, cat_dict, dict):
    """
    Update an existing fun_control dictionary with new hyperparameter values.
    All values from the hyperparameter dict (dict) are updated in the fun_control dictionary
    using the num_dict and cat_dict dictionaries.

    Args:
        fun_control (dict):
            The fun_control dictionary. This dictionary is updated with the new hyperparameter values.
        num_dict (dict):
            The dictionary containing the numerical hyperparameter values, which
            are used to update the fun_control dictionary.
        cat_dict (dict):
            The dictionary containing the categorical hyperparameter values, which
            are used to update the fun_control dictionary.
        dict (dict):
            The dictionary containing the "old" hyperparameter values.
    """
    for i, (key, value) in enumerate(dict.items()):
        if dict[key]["type"] == "int":
            set_control_hyperparameter_value(
                fun_control,
                key,
                [
                    int(num_dict[key]["lower"]),
                    int(num_dict[key]["upper"]),
                ],
            )
        if (dict[key]["type"] == "factor") and (dict[key]["core_model_parameter_type"] == "bool"):
            set_control_hyperparameter_value(
                fun_control,
                key,
                [
                    int(num_dict[key]["lower"]),
                    int(num_dict[key]["upper"]),
                ],
            )
        if dict[key]["type"] == "float":
            set_control_hyperparameter_value(
                fun_control,
                key,
                [
                    float(num_dict[key]["lower"]),
                    float(num_dict[key]["upper"]),
                ],
            )
        if dict[key]["type"] == "factor" and dict[key]["core_model_parameter_type"] != "bool":
            fle = cat_dict[key]["levels"]
            # convert the string to a list of strings
            fle = fle.split()
            set_control_hyperparameter_value(fun_control, key, fle)
            fun_control["core_model_hyper_dict"][key].update({"upper": len(fle) - 1})


def set_int_hyperparameter_values(fun_control, key, lower, upper) -> None:
    """
    Set (modify) the integer hyperparameter values in the fun_control dictionary.

    Args:
        fun_control (dict):
            The fun_control dictionary.
        key (str):
            The key of the hyperparameter.
        lower (int):
            The lower bound of the hyperparameter.
        upper (int):
            The upper bound of the hyperparameter.

    Examples:
        >>> from spotriver.hyperdict.river_hyper_dict import RiverHyperDict
            from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import set_int_hyperparameter_values
            from spotpython.utils.eda import gen_design_table
            fun_control = fun_control_init(
                core_model_name="forest.AMFRegressor",
                hyperdict=RiverHyperDict,
            )
            print("Before modification:")
            print(gen_design_table(fun_control))
            set_int_hyperparameter_values(fun_control, "n_estimators", 2, 5)
            print("After modification:")
            print(gen_design_table(fun_control))
            Seed set to 123
                Before modification:
                | name            | type   |   default |   lower |   upper | transform   |
                |-----------------|--------|-----------|---------|---------|-------------|
                | n_estimators    | int    |        10 |     2   |    1000 | None        |
                | step            | float  |         1 |     0.1 |      10 | None        |
                | use_aggregation | factor |         1 |     0   |       1 | None        |
                Setting hyperparameter n_estimators to value [2, 5].
                Variable type is int.
                Core type is None.
                Calling modify_hyper_parameter_bounds().
                After modification:
                | name            | type   |   default |   lower |   upper | transform   |
                |-----------------|--------|-----------|---------|---------|-------------|
                | n_estimators    | int    |        10 |     2   |       5 | None        |
                | step            | float  |         1 |     0.1 |      10 | None        |
                | use_aggregation | factor |         1 |     0   |       1 | None        |
    """
    set_control_hyperparameter_value(
        fun_control,
        key,
        [
            lower,
            upper,
        ],
    )


def set_float_hyperparameter_values(fun_control, key, lower, upper) -> None:
    """
    Set the float hyperparameter values in the fun_control dictionary.

    Args:
        fun_control (dict):
            The fun_control dictionary.
        key (str):
            The key of the hyperparameter.
        lower (float):
            The lower bound of the hyperparameter.
        upper (float):
            The upper bound of the hyperparameter.

    Examples:
        >>> from spotriver.hyperdict.river_hyper_dict import RiverHyperDict
            from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import set_float_hyperparameter_values
            from spotpython.utils.eda import gen_design_table
            fun_control = fun_control_init(
                core_model_name="forest.AMFRegressor",
                hyperdict=RiverHyperDict,
            )
            print("Before modification:")
            print(gen_design_table(fun_control))
            set_float_hyperparameter_values(fun_control, "step", 0.2, 5)
            print("After modification:")
            print(gen_design_table(fun_control))
            Seed set to 123
    """
    set_control_hyperparameter_value(
        fun_control,
        key,
        [
            lower,
            upper,
        ],
    )


def set_boolean_hyperparameter_values(fun_control, key, lower, upper):
    """
    Set the boolean hyperparameter values in the fun_control dictionary.

    Args:
        fun_control (dict):
            The fun_control dictionary.
        key (str):
            The key of the hyperparameter.
        lower (bool):
            The lower bound of the hyperparameter.
        upper (bool):
            The upper bound of the hyperparameter.

    Examples:
        >>> from spotriver.hyperdict.river_hyper_dict import RiverHyperDict
            from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import set_boolean_hyperparameter_values
            from spotpython.utils.eda import gen_design_table
            fun_control = fun_control_init(
                core_model_name="forest.AMFRegressor",
                hyperdict=RiverHyperDict,
            )
            print("Before modification:")
            print(gen_design_table(fun_control))
            set_boolean_hyperparameter_values(fun_control, "use_aggregation", 0, 0)
            print("After modification:")
            print(gen_design_table(fun_control))
            Seed set to 123
            Before modification:
            | name            | type   |   default |   lower |   upper | transform   |
            |-----------------|--------|-----------|---------|---------|-------------|
            | n_estimators    | int    |        10 |     2   |    1000 | None        |
            | step            | float  |         1 |     0.1 |      10 | None        |
            | use_aggregation | factor |         1 |     0   |       1 | None        |
            Setting hyperparameter use_aggregation to value [0, 0].
            Variable type is factor.
            Core type is bool.
            Calling modify_boolean_hyper_parameter_levels().
            After modification:
            | name            | type   |   default |   lower |   upper | transform   |
            |-----------------|--------|-----------|---------|---------|-------------|
            | n_estimators    | int    |        10 |     2   |    1000 | None        |
            | step            | float  |         1 |     0.1 |      10 | None        |
            | use_aggregation | factor |         1 |     0   |       0 | None        |
    """
    set_control_hyperparameter_value(
        fun_control,
        key,
        [
            lower,
            upper,
        ],
    )


def set_factor_hyperparameter_values(fun_control, key, levels):
    """
    Set the factor hyperparameter values in the fun_control dictionary.

    Args:
        fun_control (dict):
            The fun_control dictionary.
        key (str):
            The key of the hyperparameter.
        levels (list):
            The levels of the hyperparameter.

    Examples:
        >>> from spotriver.hyperdict.river_hyper_dict import RiverHyperDict
            from spotpython.utils.init import fun_control_init
            from spotpython.hyperparameters.values import set_factor_hyperparameter_values
            from spotpython.utils.eda import gen_design_table
            fun_control = fun_control_init(
                core_model_name="tree.HoeffdingTreeRegressor",
                hyperdict=RiverHyperDict,
            )
            print("Before modification:")
            print(gen_design_table(fun_control))
            set_factor_hyperparameter_values(fun_control, "leaf_model", ['LinearRegression',
                                                                'Perceptron'])
            print("After modification:")
            print(gen_design_table(fun_control))
                Seed set to 123
                Before modification:
                | name                   | type   | default          |   lower |    upper | transform              |
                |------------------------|--------|------------------|---------|----------|------------------------|
                | grace_period           | int    | 200              |  10     | 1000     | None                   |
                | max_depth              | int    | 20               |   2     |   20     | transform_power_2_int  |
                | delta                  | float  | 1e-07            |   1e-08 |    1e-06 | None                   |
                | tau                    | float  | 0.05             |   0.01  |    0.1   | None                   |
                | leaf_prediction        | factor | mean             |   0     |    2     | None                   |
                | leaf_model             | factor | LinearRegression |   0     |    2     | None                   |
                | model_selector_decay   | float  | 0.95             |   0.9   |    0.99  | None                   |
                | splitter               | factor | EBSTSplitter     |   0     |    2     | None                   |
                | min_samples_split      | int    | 5                |   2     |   10     | None                   |
                | binary_split           | factor | 0                |   0     |    1     | None                   |
                | max_size               | float  | 500.0            | 100     | 1000     | None                   |
                | memory_estimate_period | int    | 6                |   3     |    8     | transform_power_10_int |
                | stop_mem_management    | factor | 0                |   0     |    1     | None                   |
                | remove_poor_attrs      | factor | 0                |   0     |    1     | None                   |
                | merit_preprune         | factor | 1                |   0     |    1     | None                   |
                After modification:
                | name                   | type   | default          |   lower |    upper | transform              |
                |------------------------|--------|------------------|---------|----------|------------------------|
                | grace_period           | int    | 200              |  10     | 1000     | None                   |
                | max_depth              | int    | 20               |   2     |   20     | transform_power_2_int  |
                | delta                  | float  | 1e-07            |   1e-08 |    1e-06 | None                   |
                | tau                    | float  | 0.05             |   0.01  |    0.1   | None                   |
                | leaf_prediction        | factor | mean             |   0     |    2     | None                   |
                | leaf_model             | factor | LinearRegression |   0     |    1     | None                   |
                | model_selector_decay   | float  | 0.95             |   0.9   |    0.99  | None                   |
                | splitter               | factor | EBSTSplitter     |   0     |    2     | None                   |
                | min_samples_split      | int    | 5                |   2     |   10     | None                   |
                | binary_split           | factor | 0                |   0     |    1     | None                   |
                | max_size               | float  | 500.0            | 100     | 1000     | None                   |
                | memory_estimate_period | int    | 6                |   3     |    8     | transform_power_10_int |
                | stop_mem_management    | factor | 0                |   0     |    1     | None                   |
                | remove_poor_attrs      | factor | 0                |   0     |    1     | None                   |
                | merit_preprune         | factor | 1                |   0     |    1     | None                   |
    """
    # check if levels is a list of strings. If not, convert it to a list
    if not isinstance(levels, list):
        levels = [levels]
    # check if levels is a list of strings. Othewise, issue a warning and return None
    if not all(isinstance(x, str) for x in levels):
        print("!!! Warning: levels should be a list of strings.")
        return None
    # check if key "core_model_hyper_dict" exists in fun_control:
    if "core_model_hyper_dict" not in fun_control.keys():
        return None
    else:
        fun_control["core_model_hyper_dict"][key].update({"levels": levels})
        fun_control["core_model_hyper_dict"][key].update({"upper": len(levels) - 1})


def get_river_core_model_from_name(core_model_name: str) -> tuple:
    """
    Returns the river core model name and instance from a core model name.

    Args:
        core_model_name (str): The full name of the core model in the format 'module.Model'.

    Returns:
        (str, object): A tuple containing the core model name and an instance of the core model.

    Examples:
        >>> from spotpython.hyperparameters.values import get_core_model_from_name
            model_name, model_instance = get_core_model_from_name('tree.HoeffdingTreeRegressor')
            print(f"Model Name: {model_name}, Model Instance: {model_instance}")
                Model Name:
                HoeffdingTreeRegressor,
                Model Instance:
                <class 'river.tree.hoeffding_tree_regressor.HoeffdingTreeRegressor'>
    """
    # Split the model name into its components
    name_parts = core_model_name.split(".")
    if len(name_parts) < 2:
        raise ValueError(f"Invalid core model name: {core_model_name}. Expected format: 'module.ModelName'.")
    module_name = name_parts[0]
    model_name = name_parts[1]
    try:
        # Try to get the model from the river library
        core_model_instance = getattr(getattr(river, module_name), model_name)
        return model_name, core_model_instance
    except AttributeError:
        raise ValueError(f"Model '{core_model_name}' not found in either 'river' libraries.")


def get_core_model_from_name(core_model_name: str) -> tuple:
    """
    Returns the sklearn or spotpython lightning core model name and instance from a core model name.

    Args:
        core_model_name (str): The full name of the core model in the format 'module.Model'.

    Returns:
        (str, object): A tuple containing the core model name and an instance of the core model.

    Examples:
        >>> model_name, model_instance = get_core_model_from_name("light.regression.NNLinearRegressor")
            print(f"Model Name: {model_name}, Model Instance: {model_instance}")
                Model Name:
                NNLinearRegressor,
                Model Instance:
                <class 'spotpython.light.regression.nn_linear_regressor.NNLinearRegressor'>
    """
    # Split the model name into its components
    name_parts = core_model_name.split(".")
    if len(name_parts) < 2:
        raise ValueError(f"Invalid core model name: {core_model_name}. Expected format: 'module.ModelName'.")
    module_name = name_parts[0]
    model_name = name_parts[1]
    try:
        # Try to get the model from the sklearn library
        core_model_instance = getattr(getattr(sklearn, module_name), model_name)
        return model_name, core_model_instance
    except AttributeError:
        try:
            # Try to get the model from the spotpython library
            submodule_name = name_parts[1]
            model_name = name_parts[2] if len(name_parts) == 3 else model_name
            print(f"module_name: {module_name}")
            print(f"submodule_name: {submodule_name}")
            print(f"model_name: {model_name}")
            core_model_instance = getattr(getattr(getattr(spotpython, module_name), submodule_name), model_name)
            return model_name, core_model_instance
        except AttributeError:
            raise ValueError(
                f"Model '{core_model_name}' not found in either 'sklearn' or 'spotpython lightning' libraries."
            )


def get_river_prep_model(prepmodel_name) -> object:
    """
    Get the river preprocessing model from the name.

    Args:
        prepmodel_name (str): The name of the preprocessing model.

    Returns:
        river.preprocessing (object): The river preprocessing model.

    """
    if prepmodel_name == "None":
        prepmodel = None
    else:
        prepmodel = getattr(river.preprocessing, prepmodel_name)
    return prepmodel


def get_prep_model(prepmodel_name) -> object:
    """
    Get the sklearn preprocessing model from the name.

    Args:
        prepmodel_name (str): The name of the preprocessing model.

    Returns:
        river.preprocessing (object): The river preprocessing model.

    """
    if prepmodel_name == "None":
        prepmodel = None
    else:
        prepmodel = getattr(sklearn.preprocessing, prepmodel_name)
    return prepmodel


def get_sklearn_scaler(scaler_name) -> object:
    """
    Get the sklearn scaler model from the name.

    Args:
        scaler_name (str): The name of the preprocessing model.

    Returns:
        sklearn.preprocessing (object): The sklearn scaler.

    """
    if scaler_name == "None":
        scaler = None
    else:
        scaler = getattr(sklearn.preprocessing, scaler_name)
    return scaler


def get_metric_sklearn(metric_name) -> object:
    """
    Returns the sklearn metric from the metric name.

    Args:
        metric_name (str): The name of the metric.

    Returns:
        sklearn.metrics (object): The sklearn metric.
    """
    metric_sklearn = getattr(sklearn.metrics, metric_name)
    return metric_sklearn


def set_hyperparameter(fun_control, key, values):
    """
    Set hyperparameter values in the fun_control dictionary based on the type of the values argument.

    Args:
        fun_control (dict):
            The fun_control dictionary.
        key (str):
            The key of the hyperparameter.
        values (Union[int, float, bool, list]):
            The values of the hyperparameter. This can be:
                - For int and float: a list containing lower and upper bounds.
                - For bool: a list containing two boolean values.
                - For factor: a list of strings representing levels.

    Examples:
        >>> from spotpython.hyperparameters.values import set_hyperparameter
        >>> fun_control = {
                "core_model_hyper_dict": {
                    "n_estimators": {"type": "int", "default": 10, "lower": 2, "upper": 1000},
                    "step": {"type": "float", "default": 1.0, "lower": 0.1, "upper": 10.0},
                    "use_aggregation": {"type": "factor", "default": 1, "lower": 0, "upper": 1, "levels": [0, 1]},
                    "leaf_model": {"type": "factor", "default": "LinearRegression", "upper": 2}
                }
            }
        >>> set_hyperparameter(fun_control, "n_estimators", [2, 5])
        >>> set_hyperparameter(fun_control, "step", [0.2, 5.0])
        >>> set_hyperparameter(fun_control, "use_aggregation", [False, True])
        >>> set_hyperparameter(fun_control, "leaf_model", ["LinearRegression", "Perceptron"])
        >>> set_hyperparameter(fun_control, "leaf_model", "LinearRegression")
    """
    # if values is only a string  and not a list of strings, convert it to a list
    if isinstance(values, str):
        values = [values]
    if isinstance(values, list):
        if all(isinstance(v, int) for v in values):
            _set_int_hyperparameter_values(fun_control, key, values[0], values[1])
        elif all(isinstance(v, float) for v in values):
            _set_float_hyperparameter_values(fun_control, key, values[0], values[1])
        elif all(isinstance(v, bool) for v in values):
            _set_boolean_hyperparameter_values(fun_control, key, values[0], values[1])
        elif all(isinstance(v, str) for v in values):
            _set_factor_hyperparameter_values(fun_control, key, values)
        else:
            raise ValueError("Invalid type in values list.")
    else:
        raise TypeError("values should be a list.")


def _set_int_hyperparameter_values(fun_control, key, lower, upper) -> None:
    # Set integer hyperparameter values in fun_control dictionary
    fun_control["core_model_hyper_dict"][key].update({"lower": lower, "upper": upper})


def _set_float_hyperparameter_values(fun_control, key, lower, upper) -> None:
    # Set float hyperparameter values in fun_control dictionary
    fun_control["core_model_hyper_dict"][key].update({"lower": lower, "upper": upper})


def _set_boolean_hyperparameter_values(fun_control, key, lower, upper):
    # Set boolean hyperparameter values in fun_control dictionary
    fun_control["core_model_hyper_dict"][key].update({"lower": lower, "upper": upper})


def _set_factor_hyperparameter_values(fun_control, key, levels):
    # Set factor hyperparameter values in fun_control dictionary
    if "core_model_hyper_dict" not in fun_control.keys():
        return
    if not isinstance(levels, list):
        levels = [levels]
    if not all(isinstance(level, str) for level in levels):
        print("!!! Warning: levels should be a list of strings.")
        return
    fun_control["core_model_hyper_dict"][key].update({"levels": levels, "upper": len(levels) - 1})
