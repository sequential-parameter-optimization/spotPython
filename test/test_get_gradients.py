import pytest
import numpy as np
from spotpython.utils.init import fun_control_init
from spotpython.data.diabetes import Diabetes
from spotpython.light.regression.nn_linear_regressor import NNLinearRegressor
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.hyperparameters.values import (
    get_default_hyperparameters_as_array, get_one_config_from_X)
from spotpython.plot.xai import get_gradients

def test_gradients_computation():
    # Initialize the control function
    fun_control = fun_control_init(
        _L_in=10, # 10: diabetes
        _L_out=1,
        _torchmetric="mean_squared_error",
        data_set=Diabetes(),
        core_model=NNLinearRegressor,
        hyperdict=LightHyperDict
    )
    
    # Get hyperparameters and model configuration
    X = get_default_hyperparameters_as_array(fun_control)
    config = get_one_config_from_X(X, fun_control)
    
    # Retrieve specific parameters from the control
    _L_in = fun_control["_L_in"]
    _L_out = fun_control["_L_out"]
    _torchmetric = fun_control["_torchmetric"]
    batch_size = 16
    
    # Instantiate the core model with the setup configuration
    model = fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out, _torchmetric=_torchmetric)
    
    # Compute gradients using the defined function
    gradients,_ = get_gradients(
        model, 
        fun_control=fun_control, 
        batch_size=batch_size, 
        device="cpu"
    )
    
    # Conduct necessary assertions to validate gradient results
    assert isinstance(gradients, dict), "Gradients should be a dictionary."
    # Checking that all keys in gradients dictionary contain the string 'layers'
    assert all('layers' in key for key in gradients.keys()), \
        "All keys should include 'layers' in their description."
    # Ensuring all values within the gradients are numpy arrays
    assert all(isinstance(value, np.ndarray) for value in gradients.values()), \
        "All gradient values should be numpy arrays."