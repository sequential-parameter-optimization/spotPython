import pytest
import os
import pickle
import numpy as np
from spotpython.spot import Spot
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, design_control_init

def test_save_experiment(tmp_path, capsys):
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

    # Load the experiment and check its contents
    with open(filepath, "rb") as handle:
        experiment = pickle.load(handle)
        assert "design_control" in experiment, "design_control should be in the experiment dictionary."
        assert "fun_control" in experiment, "fun_control should be in the experiment dictionary."
        assert "optimizer_control" in experiment, "optimizer_control should be in the experiment dictionary."
        assert "spot_tuner" in experiment, "spot_tuner should be in the experiment dictionary."
        assert "surrogate_control" in experiment, "surrogate_control should be in the experiment dictionary."

    # Test overwrite functionality
    S.save_experiment(filename=filename, path=path, overwrite=False)
    captured = capsys.readouterr()
    assert "Error: File" in captured.out
    assert "already exists. Use overwrite=True to overwrite the file." in captured.out

if __name__ == "__main__":
    pytest.main()