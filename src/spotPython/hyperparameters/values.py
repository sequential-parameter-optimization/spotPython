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


def modify_hyper_parameter_levels(fun_control, hyperparameter, levels) -> dict:
    """
    This function modifies the levels of a hyperparameter in the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary
        hyperparameter (str):
            hyperparameter name
        levels (list):
            list of levels

    Returns:
        fun_control (dict):
            updated fun_control

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


def modify_hyper_parameter_bounds(fun_control, hyperparameter, bounds) -> dict:
    """
    Args:
        fun_control (dict):
            fun_control dictionary
        hyperparameter (str):
            hyperparameter name
        bounds (list):
            list of two bound values. The first value represents the lower bound
            and the second value represents the upper bound.

    Returns:
        fun_control (dict):
            updated fun_control

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


def add_core_model_to_fun_control(core_model, fun_control, hyper_dict=None, filename=None) -> dict:
    """Add the core model to the function control dictionary.

    Args:
        core_model (class):
            The core model.
        fun_control (dict):
            The fun_control dictionary.
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

    Examples:
        >>> from spotPython.light.netlightregressione import NetLightRegression
            from spotPython.hyperdict.light_hyper_dict import LightHyperDict
            from spotPython.hyperparameters.values import add_core_model_to_fun_control
            add_core_model_to_fun_control(core_model=NetLightRegression,
                              fun_control=fun_control,
                              hyper_dict=LightHyperDict)
            # or, if a user wants to use a custom hyper_dict:
        >>> from spotPython.light.netlightregression import NetLightRegression
            from spotPython.hyperparameters.values import add_core_model_to_fun_control
            add_core_model_to_fun_control(core_model=NetLightRegression,
                                        fun_control=fun_control,
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
    fun_control.update({"var_type": var_type, "var_name": var_name})


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


def set_data_set(fun_control, data_set) -> dict:
    """
    This function sets the lightning dataset in the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary
        data_set (class): Dataset class from torch.utils.data

    Returns:
        fun_control (dict):
            updated fun_control

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import set_data_module
            from spotPython.data.lightdatamodule import LightDataModule
            from spotPython.data.csvdataset import CSVDataset
            from spotPython.data.pkldataset import PKLDataset
            import torch
            fun_control = fun_control_init()
            ds = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
            set_data_set(fun_control=fun_control,
                         data_set=ds)
            fun_control["data_set"]
    """
    fun_control.update({"data_set": data_set})


def set_data_module(fun_control, data_module) -> dict:
    """
    This function sets the lightning datamodule in the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary
        data_module (class): DataLoader class from torch.utils.data

    Returns:
        fun_control (dict):
            updated fun_control

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import set_data_module
            from spotPython.data.lightdatamodule import LightDataModule
            from spotPython.data.csvdataset import CSVDataset
            from spotPython.data.pkldataset import PKLDataset
            import torch
            fun_control = fun_control_init()
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=torch.long)
            dm = LightDataModule(dataset=dataset, batch_size=5, test_size=7)
            dm.setup()
            set_data_module(fun_control=fun_control,
                            data_module=dm)
            fun_control["data_module"]
    """
    fun_control.update({"data_module": data_module})


def get_tuned_architecture(spot_tuner, fun_control) -> dict:
    """
    Returns the tuned architecture.

    Args:
        spot_tuner (object):
            spot tuner object.
        fun_control (dict):
            dictionary containing control parameters for the hyperparameter tuning.

    Returns:
        (dict):
            dictionary containing the tuned architecture.
    """
    X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1, -1))
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


def set_fun_control_fun_evals(fun_control, fun_evals) -> dict:
    """
    This function sets the number of function evaluations in the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary
        fun_evals (int): number of function evaluations

    Returns:
        fun_control (dict):
            updated fun_control

    Attributes:
        fun_evals (int): number of function evaluations

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import set_fun_control_fun_evals
            fun_control = fun_control_init()
            set_fun_control_fun_evals(fun_control=fun_control,
                          fun_evals=5)
            fun_control["fun_evals"]

    """
    fun_control.update({"fun_evals": fun_evals})


def get_fun_control_fun_evals(fun_control=None) -> int:
    """
    This function gets the number of function evaluations from the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary

    Returns:
        fun_evals (int):
            number of function evaluations

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import get_fun_control_fun_evals
            fun_control = fun_control_init()
            get_fun_control_fun_evals(fun_control=fun_control)
            0
    """
    # check if key "fun_evals" exists in fun_control:
    if fun_control is None or "fun_evals" not in fun_control.keys():
        return None
    else:
        return fun_control["fun_evals"]


def set_fun_control_fun_repeats(fun_control, fun_repeats) -> dict:
    """
    This function sets the number of function repeats in the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary
        fun_repeats (int): number of function repeats

    Returns:
        fun_control (dict):
            updated fun_control

    Attributes:
        fun_repeats (int): number of function repeats

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import set_fun_control_fun_repeats
            fun_control = fun_control_init()
            set_fun_control_fun_repeats(fun_control=fun_control,
                          fun_repeats=5)
            fun_control["fun_repeats"]

    """
    fun_control.update({"fun_repeats": fun_repeats})


def get_fun_control_fun_repeats(fun_control=None) -> int:
    """
    This function gets the number of function repeats from the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary

    Returns:
        fun_repeats (int):
            number of function repeats

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import get_fun_control_fun_repeats
            fun_control = fun_control_init()
            get_fun_control_fun_repeats(fun_control=fun_control)
            0
    """
    # check if key "fun_repeats" exists in fun_control:
    if fun_control is None or "fun_repeats" not in fun_control.keys():
        return None
    else:
        return fun_control["fun_repeats"]


def set_fun_control_seed(fun_control, seed) -> dict:
    """
    This function sets the seed in the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary
        seed (int): seed

    Returns:
        fun_control (dict):
            updated fun_control

    Attributes:
        seed (int): seed

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import set_fun_control_seed
            fun_control = fun_control_init()
            set_fun_control_seed(fun_control=fun_control,
                          seed=5)
            fun_control["seed"]

    """
    fun_control.update({"seed": seed})


def get_fun_control_seed(fun_control=None) -> int:
    """
    This function gets the seed from the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary

    Returns:
        seed (int):
            seed

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import get_fun_control_seed
            fun_control = fun_control_init()
            get_fun_control_seed(fun_control=fun_control)
            0
    """
    # check if key "seed" exists in fun_control:
    if fun_control is None or "seed" not in fun_control.keys():
        return None
    else:
        return fun_control["seed"]


def set_fun_control_sigma(fun_control, sigma) -> dict:
    """
    This function sets the sigma in the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary
        sigma (float): sigma

    Returns:
        fun_control (dict):
            updated fun_control

    Attributes:
        sigma (float): sigma

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import set_fun_control_sigma
            fun_control = fun_control_init()
            set_fun_control_sigma(fun_control=fun_control,
                          sigma=5)
            fun_control["sigma"]

    """
    fun_control.update({"sigma": sigma})


def get_fun_control_sigma(fun_control=None) -> float:
    """
    This function gets the sigma from the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary

    Returns:
        sigma (float):
            sigma

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import get_fun_control_sigma
            fun_control = fun_control_init()
            get_fun_control_sigma(fun_control=fun_control)
            0
    """
    # check if key "sigma" exists in fun_control:
    if fun_control is None or "sigma" not in fun_control.keys():
        return None
    else:
        return fun_control["sigma"]


def set_fun_control_max_time(fun_control, max_time) -> dict:
    """
    This function sets the max_time in the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary
        max_time (float): max_time

    Returns:
        fun_control (dict):
            updated fun_control

    Attributes:
        max_time (float): max_time

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import set_fun_control_max_time
            fun_control = fun_control_init()
            set_fun_control_max_time(fun_control=fun_control,
                          max_time=5)
            fun_control["max_time"]

    """
    fun_control.update({"max_time": max_time})


def get_fun_control_max_time(fun_control=None) -> float:
    """
    This function gets the max_time from the fun_control dictionary.

    Args:
        fun_control (dict):
            fun_control dictionary

    Returns:
        max_time (float):
            max_time

    Examples:
        >>> from spotPython.utils.init import fun_control_init
            from spotPython.hyperparameters.values import get_fun_control_max_time
            fun_control = fun_control_init()
            get_fun_control_max_time(fun_control=fun_control)
            0
    """
    # check if key "max_time" exists in fun_control:
    if fun_control is None or "max_time" not in fun_control.keys():
        return None
    else:
        return fun_control["max_time"]
