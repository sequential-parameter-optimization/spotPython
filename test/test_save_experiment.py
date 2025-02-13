import pytest
import os
import pickle
import numpy as np
from spotpython.spot import Spot
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, design_control_init
from spotpython.utils.file import load_experiment, load_result

def test_save_experiment(tmp_path, capsys):
    PREFIX="test_save_experiment"
    
    # Initialize function control
    fun_control = fun_control_init(
        save_experiment=True,
        PREFIX=PREFIX,
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
    
    Sexp_load = load_experiment(PREFIX)
    assert Sexp_load.design_control["init_size"] == 7

    # Run the optimization to generate some data
    X_start = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    S.run(X_start=X_start)
    
    Srun_load = load_result(PREFIX)
    assert Srun_load.design_control["init_size"] == 7
    assert Srun_load.X.shape[0] > 0
    assert Srun_load.y.shape[0] > 0



if __name__ == "__main__":
    pytest.main()