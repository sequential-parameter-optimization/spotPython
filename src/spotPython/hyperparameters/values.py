import numpy as np
import copy
import json
from sklearn.pipeline import make_pipeline
from river import compose
from typing import Union, List, Dict, Generator, Any

from spotPython.utils.convert import class_for_name
from spotPython.utils.transform import transform_hyper_parameter_values


def generate_one_config_from_var_dict(
    var_dict: Dict[str, np.ndarray], fun_control: Dict[str, Union[List[str], str]]
) -> Generator[Dict[str, Union[int, float]], None, None]:
    """Generate one configuration from a dictionary of variables (as a generator).

    This function takes a dictionary of variables as input arguments and returns a generator
    that yields dictionaries with the values from the arrays in the input dictionary.

    Args:
        var_dict (dict): A dictionary where keys are variable names and values are numpy arrays.
        fun_control (dict): A dictionary which (at least) has an entry with the following key:
            "var_type" (list): A list of variable types. If the entry is not "num" the corresponding
            value will be converted to the type "int".

    Returns:
        Generator[dict]: A generator that yields dictionaries with the values from the arrays in the input dictionary.

    Examples:
        >>> import numpy as np
        >>> from spotPython.hyperparameters.values import generate_one_config_from_var_dict
        >>> var_dict = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}
        >>> fun_control = {"var_type": ["int", "num"]}
        >>> list(generate_one_config_from_var_dict(var_dict, fun_control))
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
    """
    for values in iterate_dict_values(var_dict):
        values = convert_keys(values, fun_control["var_type"])
        values = get_dict_with_levels_and_types(fun_control=fun_control, v=values)
        values = transform_hyper_parameter_values(fun_control=fun_control, hyper_parameter_values=values)
        yield values


def return_conf_list_from_var_dict(
    var_dict: Dict[str, np.ndarray], fun_control: Dict[str, Union[List[str], str]]
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
        >>> from spotPython.hyperparameters.values import return_conf_list_from_var_dict
        >>> var_dict = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}
        >>> fun_control = {'var_type': ['int', 'int']}
        >>> return_conf_list_from_var_dict(var_dict, fun_control)
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
    """
    conf_list = []
    for values in generate_one_config_from_var_dict(var_dict, fun_control):
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
        >>> from spotPython.hyperparameters.values import iterate_dict_values
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
        >>> from spotPython.hyperparameters.values import convert_keys
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


def get_dict_with_levels_and_types(fun_control: Dict[str, Any], v: Dict[str, Any]) -> Dict[str, Any]:
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
        fun_control (Dict[str, Any]): A dictionary containing information about the core model hyperparameters.
        v (Dict[str, Any]): A dictionary containing the numerical output of the hyperparameter optimization.

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
        >>> from spotPython.hyperparameters.values import assign_values
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
            from spotPython.hyperparameters.values import modify_hyper_parameter_levels
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
        >>> from spotPython.hyperparameters.values import modify_hyper_parameter_levels
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
        >>> from spotPython.hyperparameters.values import get_default_values
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
        >>> from spotPython.hyperparameters.values import get_var_type
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
        >>> from spotPython.hyperparameters.values import get_transform
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
        >>> from spotPython.hyperparameters.values import get_var_name
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
        >>> from spotPython.hyperparameters.values import get_bound_values
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
        >>> from spotPython.hyperparameters.values import replace_levels_with_positions
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
    for key, value in hyper_dict_values.items():
        if key in hyper_dict.keys():
            if "levels" in hyper_dict[key].keys():
                hyper_dict_values_new[key] = hyper_dict[key]["levels"].index(value)
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
        >>> from spotPython.hyperparameters.values import get_values_from_dict
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
        "var_type": A list of variable types.
        "var_name": A list of variable names.
        The original hyperparameters of the core model are stored in the "core_model_hyper_dict" key.

    Examples:
        >>> from spotPython.light.regression.netlightregression import NetLightRegression
            from spotPython.hyperdict.light_hyper_dict import LightHyperDict
            from spotPython.hyperparameters.values import add_core_model_to_fun_control
            add_core_model_to_fun_control(fun_control=fun_control,
                                        core_model=NetLightRegression,
                                        hyper_dict=LightHyperDict)
            # or, if a user wants to use a custom hyper_dict:
        >>> from spotPython.light.regression.netlightregression import NetLightRegression
            from spotPython.hyperparameters.values import add_core_model_to_fun_control
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
    var_type = get_var_type(fun_control)
    var_name = get_var_name(fun_control)
    lower = get_bound_values(fun_control, "lower", as_list=False)
    upper = get_bound_values(fun_control, "upper", as_list=False)
    fun_control.update({"var_type": var_type, "var_name": var_name, "lower": lower, "upper": upper})


def get_one_core_model_from_X(X, fun_control=None):
    """Get one core model from X.

    Args:
        X (np.array):
            The array with the hyper parameter values.
        fun_control (dict):
            The function control dictionary.

    Returns:
        (class):
            The core model.

    Examples:
        >>> from river.tree import HoeffdingAdaptiveTreeRegressor
            from spotRiver.data.river_hyper_dict import RiverHyperDict
            fun_control = {}
            add_core_model_to_fun_control(core_model=HoeffdingAdaptiveTreeRegressor,
                fun_control=func_control,
                hyper_dict=RiverHyperDict,
                filename=None)
            X = np.array([0, 0, 0, 0, 0])
            get_one_core_model_from_X(X, fun_control)
            HoeffdingAdaptiveTreeRegressor()
    """
    var_dict = assign_values(X, fun_control["var_name"])
    config = return_conf_list_from_var_dict(var_dict, fun_control)[0]
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
            from spotRiver.data.river_hyper_dict import RiverHyperDict
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
            from spotRiver.data.sklearn_hyper_dict import SklearnHyperDict
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
            from spotRiver.data.river_hyper_dict import RiverHyperDict
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
            from spotRiver.data.river_hyper_dict import RiverHyperDict
            from spotPython.hyperparameters.values import (
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
    X0 = replace_levels_with_positions(fun_control["core_model_hyper_dict"], X0)
    X0 = get_values_from_dict(X0)
    X0 = np.array([X0])
    X0.shape[1]
    return X0


def get_default_hyperparameters_for_core_model(fun_control) -> dict:
    """Get the default hyper parameters for the core model.

    Args:
        fun_control (dict):
            The function control dictionary.

    Returns:
        (dict):
            The default hyper parameters for the core model.

    Examples:
        >>> from river.tree import HoeffdingAdaptiveTreeRegressor
            from spotRiver.data.river_hyper_dict import RiverHyperDict
            fun_control = {}
            add_core_model_to_fun_control(core_model=HoeffdingAdaptiveTreeRegressor,
                fun_control=func_control,
                hyper_dict=RiverHyperDict,
                filename=None)
            get_default_hyperparameters_for_core_model(fun_control)
            {'leaf_prediction': 'mean',
            'leaf_model': 'NBAdaptive',
            'splitter': 'HoeffdingAdaptiveTreeSplitter',
            'binary_split': 'info_gain',
            'stop_mem_management': False}
    """
    values = get_default_values(fun_control)
    values = get_dict_with_levels_and_types(fun_control=fun_control, v=values)
    values = convert_keys(values, fun_control["var_type"])
    values = transform_hyper_parameter_values(fun_control=fun_control, hyper_parameter_values=values)
    return values


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
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import set_control_key_value
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
    vt = get_var_type_from_var_name(fun_control=control_dict, var_name=hyperparameter)
    if vt == "factor":
        modify_hyper_parameter_levels(fun_control=control_dict, hyperparameter=hyperparameter, levels=value)
    else:
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
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import get_control_key_value
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
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import get_var_type_from_var_name
            control_dict = fun_control_init()
            get_var_type_from_var_name(var_name="max_depth",
                            fun_control=control_dict)
            "int"
    """
    var_type_list = get_control_key_value(control_dict=fun_control, key="var_type")
    var_name_list = get_control_key_value(control_dict=fun_control, key="var_name")
    return var_type_list[var_name_list.index(var_name)]


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
        >>> from spotPython.utils.device import getDevice
            from spotPython.utils.init import fun_control_init
            from spotPython.utils.file import get_experiment_name, get_spot_tensorboard_path
            import numpy as np
            from spotPython.data.diabetes import Diabetes
            from spotPython.light.regression.netlightregression import NetLightRegression
            from spotPython.hyperdict.light_hyper_dict import LightHyperDict
            from spotPython.hyperparameters.values import add_core_model_to_fun_control
            from spotPython.hyperparameters.values import get_ith_hyperparameter_name_from_fun_control
            from spotPython.hyperparameters.values import set_control_key_value
            from spotPython.hyperparameters.values import set_control_hyperparameter_value
            experiment_name = get_experiment_name(prefix="000")
            fun_control = fun_control_init(
                spot_tensorboard_path=get_spot_tensorboard_path(experiment_name),
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
