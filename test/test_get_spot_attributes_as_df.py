import pytest
import numpy as np
import pandas as pd
from math import inf
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def test_get_spot_attributes_as_df():
    # Setup: Configure initial parameters
    ni = 7
    n = 10
    fun = Analytical().fun_sphere
    fun_control = fun_control_init(
        PREFIX= "test_get_spot_attributes_as_df",
        lower=np.array([-1]),
        upper=np.array([1]),
        fun_evals=n
    )
    design_control = design_control_init(init_size=ni)

    # Create instance of the Spot class
    S = Spot(
        fun=fun,
        fun_control=fun_control,
        design_control=design_control
    )

    # Run the optimization
    S.run()

    # Get the attributes as a DataFrame
    df = S.get_spot_attributes_as_df()

    # Define expected attribute names (ensure these match your Spot class' attributes)
    expected_attributes = ['X',
                            'all_lower',
                            'all_upper',
                            'all_var_name',
                            'all_var_type',
                            'counter',
                            'de_bounds',
                            'design',
                            'design_control',
                            'eps',
                            'fun_control',
                            'fun_evals',
                            'fun_repeats',
                            'ident',
                            'infill_criterion',
                            'k',
                            'log_level',
                            'lower',
                            'max_surrogate_points',
                            'max_time',
                            'mean_X',
                            'mean_y',
                            'min_X',
                            'min_mean_X',
                            'min_mean_y',
                            'min_y',
                            'n_points',
                            'noise',
                            'ocba_delta',
                            'optimizer_control',
                            'progress_file',
                            'red_dim',
                            'rng',
                            'show_models',
                            'show_progress',
                            'spot_writer',
                            'surrogate',
                            'surrogate_control',
                            'tkagg',
                            'tolerance_x',
                            'upper',
                            'var_name',
                            'var_type',
                            'var_y',
                            'verbosity',
                            'y',
                            'y_mo']

    # Check that the DataFrame has the correct attributes
    assert list(df['Attribute Name']) == expected_attributes

    # Further checks can be done for specific attribute values
    # Example: Check that 'fun_evals' has the expected value
    fun_evals_row = df.query("`Attribute Name` == 'fun_evals'")
    assert not fun_evals_row.empty and fun_evals_row['Attribute Value'].values[0] == n

    # Example: Check that 'lower' has the expected value
    lower_row = df.query("`Attribute Name` == 'lower'")
    assert not lower_row.empty and lower_row['Attribute Value'].values[0] == [-1]