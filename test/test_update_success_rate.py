import numpy as np
import pytest
from spotpython.spot.spot import Spot

@pytest.fixture
def spot_instance():
    # Minimal Spot instance with required attributes
    fun_control = {"lower": np.array([-1, -1]), "upper": np.array([1, 1]), "var_type": ["num", "num"], "fun_evals": 10, "fun_repeats": 1, "max_time": 1e6, "noise": False, "tolerance_x": 0, "ocba_delta": 0, "log_level": 50, "show_models": False, "show_progress": False, "infill_criterion": "ei", "n_points": 1, "seed": 1, "progress_file": None, "tkagg": False, "verbosity": 0, "acquisition_failure_strategy": "random", "fun_mo2so": None, "penalty_NA": 0, "PREFIX": "test", "tensorboard_log": False, "spot_tensorboard_path": None, "save_experiment": False, "db_dict_name": None, "save_result": False, "var_name": ["x0", "x1"]}
    design_control = {"init_size": 5, "repeats": 1}
    from spotpython.utils.init import surrogate_control_init, optimizer_control_init
    surrogate_control = surrogate_control_init()
    optimizer_control = optimizer_control_init()
    S = Spot(fun=lambda X, fun_control=None: np.sum(X, axis=1), fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control, optimizer_control=optimizer_control)
    S.window_size = 5
    S.y = np.array([5, 4, 3, 2, 1])
    return S

def test_update_success_rate_basic(spot_instance):
    S = spot_instance
    # Add new values, only the last value is a new minimum
    y_new = np.array([6, 0])
    S._update_success_rate(y_new)
    # Only the last value (0) is a success, so last 5 successes: [1] (from S.y), [1] (from 0), rest 0
    # S.y: [5,4,3,2,1] -> best_y = 1, [6,0] -> 6>1 (fail), 0<1 (success)
    # So _success_history should end with [1,0,1] (last 5: [1,0,1])
    assert hasattr(S, "success_rate")
    assert S.success_rate == sum(S._success_history) / len(S._success_history)
    assert S._success_history[-1] == 1  # last was a success

def test_update_success_rate_all_failures(spot_instance):
    S = spot_instance
    # Add new values, none are new minima
    y_new = np.array([6, 7])
    S._update_success_rate(y_new)
    # With the new logic, only new minima are counted as successes.
    # So after adding [6, 7], no new successes, so success_rate == 0.0
    assert S.success_rate == 0.0
    assert S._success_history[-2:] == [0, 0]

def test_update_success_rate_all_successes(spot_instance):
    S = spot_instance
    # Add new values, all are new minima
    y_new = np.array([-1, -2])
    S._update_success_rate(y_new)
    # S.y: [5,4,3,2,1] -> best_y = 1, -1 < 1 (success), best_y = -1, -2 < -1 (success)
    # So last 5: [3,2,1,-1,-2] -> all are new minima, so all successes
    assert S.success_rate == 1.0
    assert all(x == 1 for x in S._success_history)

def test_update_success_rate_window_size(spot_instance):
    S = spot_instance
    # Add enough new values to exceed window_size
    y_new = np.arange(10, 0, -1)
    S._update_success_rate(y_new)
    # The history should be of length window_size
    assert len(S._success_history) == S.window_size