import os
import numpy as np
import pytest
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

# Define a top-level, picklable test function
def quadratic_fun(X, **kwargs):
    return np.sum(X**2, axis=1)

@pytest.fixture
def spot_instance(tmp_path):
    PREFIX = "pytest_save_experiment"
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        fun_evals=5,
        PREFIX=PREFIX,
        save_result=False,
        save_experiment=False,
    )
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=quadratic_fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design()
    spot.evaluate_initial_design()
    spot.update_stats()
    spot.fit_surrogate()
    spot.update_design()
    # Change working directory for this test
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield spot
    os.chdir(old_cwd)

def test_save_experiment_creates_file(spot_instance):
    spot = spot_instance
    spot.save_experiment()
    expected_file = "pytest_save_experiment_exp.pkl"
    assert os.path.exists(expected_file)

def test_save_experiment_custom_filename(spot_instance, tmp_path):
    spot = spot_instance
    custom_file = tmp_path / "custom_exp.pkl"
    spot.save_experiment(filename=str(custom_file))
    assert custom_file.exists()

def test_save_experiment_with_path(spot_instance, tmp_path):
    spot = spot_instance
    subdir = tmp_path / "results"
    spot.save_experiment(path=str(subdir))
    expected_file = subdir / "pytest_save_experiment_exp.pkl"
    assert expected_file.exists()

def test_save_experiment_overwrite_false(tmp_path):
    PREFIX = "pytest_save_experiment2"
    fun_control = fun_control_init(
        lower=np.array([-1, -1]),
        upper=np.array([1, 1]),
        fun_evals=5,
        PREFIX=PREFIX,
        save_result=False,
        save_experiment=False,
    )
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=quadratic_fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design()
    spot.evaluate_initial_design()
    spot.update_stats()
    spot.fit_surrogate()
    spot.update_design()
    os.chdir(tmp_path)
    filename = "pytest_save_experiment2_exp.pkl"
    # Create file first
    with open(filename, "wb") as f:
        f.write(b"dummy")
    # Should not overwrite
    spot.save_experiment(filename=filename, overwrite=False)
    with open(filename, "rb") as f:
        content = f.read()
    assert content == b"dummy"
    os.chdir("..")