import pytest
import os
import pickle
import numpy as np
from spotpython.spot import Spot
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, design_control_init
from spotpython.utils.file import load_experiment

def compare_dicts(dict1, dict2):
    """
    Compare two dictionaries, including element-wise comparison for numpy arrays.
    """
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if isinstance(dict1[key], np.ndarray) and isinstance(dict2[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    return True

def test_save_and_load_experiment(tmp_path):
    # Initialize function control
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1])
    )

    # Initialize design control
    design_control = design_control_init(init_size=7)

    # Define the objective function
    fun = Analytical().fun_sphere

    # Create the Spot instance
    S = Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control,
    )

    # Run the optimization to generate some data
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    S.run(X_start=X_start)

    # Define the filename and path
    filename = "test_experiment.pkl"
    path = tmp_path

    # Save the experiment
    S.save_experiment(filename=filename, path=path)

    # Check if the file was created
    filepath = os.path.join(path, filename)
    assert os.path.exists(filepath), f"File {filepath} should exist."

    # Load the experiment
    spot_tuner, loaded_fun_control, loaded_design_control, loaded_surrogate_control, loaded_optimizer_control = load_experiment(PKL_NAME=filepath)

    # Check if the loaded data matches the original data
    assert compare_dicts(loaded_fun_control, fun_control), "Loaded fun_control should match the original fun_control."
    assert compare_dicts(loaded_design_control, design_control), "Loaded design_control should match the original design_control."
    assert compare_dicts(loaded_surrogate_control, S.surrogate_control), "Loaded surrogate_control should match the original surrogate_control."
    assert compare_dicts(loaded_optimizer_control, S.optimizer_control), "Loaded optimizer_control should match the original optimizer_control."

    # Check if the spot_tuner is an instance of Spot
    assert isinstance(spot_tuner, Spot), "Loaded spot_tuner should be an instance of Spot."

    # Check if the design matrix and response vector are equal
    assert np.array_equal(spot_tuner.X, S.X), "Loaded design matrix should match the original design matrix."
    assert np.array_equal(spot_tuner.y, S.y), "Loaded response vector should match the original response vector."

if __name__ == "__main__":
    pytest.main()