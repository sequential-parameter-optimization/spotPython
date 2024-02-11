import os
from spotPython.utils.file import save_experiment, load_experiment
import numpy as np
from math import inf
from spotPython.spot import spot
from spotPython.utils.init import (
    fun_control_init,
    design_control_init,
    surrogate_control_init,
    optimizer_control_init)
from spotPython.fun.objectivefunctions import analytical

def test_file_save_load():
    fun = analytical().fun_branin

    fun_control = fun_control_init(
                PREFIX="branin",
                SUMMARY_WRITER=False,
                lower = np.array([0, 0]),
                upper = np.array([10, 10]),
                fun_evals=8,
                fun_repeats=1,
                max_time=inf,
                noise=False,
                tolerance_x=0,
                ocba_delta=0,
                var_type=["num", "num"],
                infill_criterion="ei",
                n_points=1,
                seed=123,
                log_level=20,
                show_models=False,
                show_progress=True)
    design_control = design_control_init(
                init_size=5,
                repeats=1)
    surrogate_control = surrogate_control_init(
                model_fun_evals=10000,
                min_theta=-3,
                max_theta=3,
                n_theta=2,
                theta_init_zero=True,
                n_p=1,
                optim_p=False,
                var_type=["num", "num"],
                seed=124)
    optimizer_control = optimizer_control_init(
                max_iter=1000,
                seed=125)
    spot_tuner = spot.Spot(fun=fun,
                fun_control=fun_control,
                design_control=design_control,
                surrogate_control=surrogate_control,
                optimizer_control=optimizer_control)
    # Call the save_experiment function
    pkl_name = save_experiment(
        spot_tuner=spot_tuner,
        fun_control=fun_control,
        design_control=None,
        surrogate_control=None,
        optimizer_control=None
    )

    # Verify that the pickle file is created
    assert os.path.exists(pkl_name)

    # Call the load_experiment function
    spot_tuner_1, fun_control_1, design_control_1, surrogate_control_1, optimizer_control_1 = load_experiment(pkl_name)

    # Verify the name of the pickle file
    assert pkl_name == f"spot_{fun_control['PREFIX']}_experiment.pickle"

    # Clean up the temporary directory
    os.remove(pkl_name)