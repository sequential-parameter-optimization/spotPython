import pytest
import numpy as np
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, surrogate_control_init

def dummy_fun(X, **kwargs):
    return np.sum(X, axis=1)

def test_set_additional_attributes_basic():
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        fun_evals=5,
        fun_repeats=2,
        max_time=10,
        noise=True,
        tolerance_x=0.1,
        ocba_delta=2,
        log_level=30,
        show_models=True,
        show_progress=False,
        infill_criterion="ei",
        n_points=3,
        progress_file="progress.txt",
        tkagg=False,
        min_success_rate=0.5,
        verbosity=2,
        acquisition_failure_strategy="random"
    )
    surrogate_control = surrogate_control_init(
        max_surrogate_points=50,
        use_nystrom=True,
        nystrom_m=10,
        nystrom_seed=42
    )
    spot = Spot(fun=dummy_fun, fun_control=fun_control, surrogate_control=surrogate_control)
    spot._set_additional_attributes()
    assert spot.fun_evals == 5
    assert spot.fun_repeats == 2
    assert spot.max_time == 10
    assert spot.noise is True
    assert spot.tolerance_x == 0.1
    assert spot.ocba_delta == 2
    assert spot.log_level == 30
    assert spot.show_models is True
    assert spot.show_progress is False
    assert spot.infill_criterion == "ei"
    assert spot.n_points == 3
    assert spot.progress_file == "progress.txt"
    assert spot.tkagg is False
    assert spot.min_success_rate == 0.5
    assert spot.verbosity == 2
    assert spot.acquisition_failure_strategy == "random"
    assert spot.max_surrogate_points == 50
    assert spot.use_nystrom is True
    assert spot.nystrom_m == 10
    assert spot.nystrom_seed == 42

def test_set_additional_attributes_defaults():
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1])
    )
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    spot._set_additional_attributes()
    # Check that attributes exist and are set to their default values
    assert hasattr(spot, "fun_evals")
    assert hasattr(spot, "fun_repeats")
    assert hasattr(spot, "max_time")
    assert hasattr(spot, "noise")
    assert hasattr(spot, "tolerance_x")
    assert hasattr(spot, "ocba_delta")
    assert hasattr(spot, "log_level")
    assert hasattr(spot, "show_models")
    assert hasattr(spot, "show_progress")
    assert hasattr(spot, "infill_criterion")
    assert hasattr(spot, "n_points")
    assert hasattr(spot, "progress_file")
    assert hasattr(spot, "tkagg")
    assert hasattr(spot, "min_success_rate")
    assert hasattr(spot, "verbosity")
    assert hasattr(spot, "acquisition_failure_strategy")
    assert hasattr(spot, "max_surrogate_points")
    assert hasattr(spot, "use_nystrom")
    assert hasattr(spot, "nystrom_m")
    assert hasattr(spot, "nystrom_seed")
