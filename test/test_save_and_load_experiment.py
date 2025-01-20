import pytest
import pprint
from spotpython.spot import Spot
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, design_control_init
from spotpython.utils.file import load_experiment
import numpy as np

def _compare_dicts(dict1, dict2, ignore_keys=None):
    """
    Compare two dictionaries, including element-wise comparison for numpy arrays.
    Print missing elements (keys) if the dictionaries do not match.

    Args:
        dict1 (dict): First dictionary to compare.
        dict2 (dict): Second dictionary to compare.
        ignore_keys (list, optional): List of keys to ignore during comparison. Default is None.

    Returns:
        bool: True if the dictionaries match, False otherwise.
    """
    if ignore_keys is None:
        ignore_keys = []
    # ensure that ignore_keys is a list
    if not isinstance(ignore_keys, list):
        ignore_keys = [ignore_keys]

    keys1 = set(dict1.keys()) - set(ignore_keys)
    keys2 = set(dict2.keys()) - set(ignore_keys)

    if keys1 != keys2:
        missing_in_dict1 = keys2 - keys1
        missing_in_dict2 = keys1 - keys2
        print(f"Missing in dict1: {missing_in_dict1}")
        print(f"Missing in dict2: {missing_in_dict2}")
        return False

    for key in keys1:
        if isinstance(dict1[key], np.ndarray) and isinstance(dict2[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                print(f"Mismatch in key '{key}': {dict1[key]} != {dict2[key]}")
                return False
        else:
            if dict1[key] != dict2[key]:
                print(f"Mismatch in key '{key}': {dict1[key]} != {dict2[key]}")
                return False

    return True

def test_save_and_load_experiment():
    PREFIX = "test_save_and_load_experiment_03"
    # Initialize function control
    fun_control = fun_control_init(
        save_experiment=True,
        PREFIX=PREFIX,
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        verbosity=1
    )
    
    design_control = design_control_init(init_size=7)

    fun = Analytical().fun_sphere
        
    S = Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
    )

    # Load the experiment
    S_loaded = load_experiment(PREFIX)
    print(f"S: {S}")    
    print(f"S_loaded: {S_loaded}")
    pprint.pprint(S_loaded)
    loaded_fun_control = S_loaded.fun_control
    loaded_design_control = S_loaded.design_control
    loaded_surrogate_control = S_loaded.surrogate_control
    loaded_optimizer_control = S_loaded.optimizer_control
    
    # Check if the loaded data matches the original data
    # It is ok if the counter is different, because it is increased during the run
    assert _compare_dicts(loaded_fun_control, fun_control, ignore_keys="counter"), "Loaded fun_control should match the original fun_control."
    assert _compare_dicts(loaded_design_control, design_control), "Loaded design_control should match the original design_control."
    assert _compare_dicts(loaded_surrogate_control, S.surrogate_control), "Loaded surrogate_control should match the original surrogate_control."
    assert _compare_dicts(loaded_optimizer_control, S.optimizer_control), "Loaded optimizer_control should match the original optimizer_control."

    # Check if the S_loaded is an instance of Spot
    assert isinstance(S_loaded, Spot), "Loaded S_loaded should be an instance of Spot."

    # Check if the design matrix and response vector are equal
    # if there are differences, print the differences
    # Differences are OK
    # if not np.array_equal(S_loaded.X, S.X):
    #     print(f"Design matrix mismatch: {S_loaded.X} != {S.X}")
    # if not np.array_equal(S_loaded.y, S.y):
    #     print(f"Response vector mismatch: {S_loaded.y} != {S.y}")

if __name__ == "__main__":
    pytest.main()