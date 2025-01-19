from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.hyperparameters.values import get_core_model_from_name, add_core_model_to_fun_control, generate_one_config_from_var_dict
import numpy as np


def test_generate_one_config_from_var_dict():
    core_model_name="light.regression.NNLinearRegressor"
    hyperdict=LightHyperDict
    fun_control = {}
    _ , core_model_instance = get_core_model_from_name(core_model_name)
    add_core_model_to_fun_control(
        core_model=core_model_instance,
        fun_control=fun_control,
        hyper_dict=hyperdict,
        filename=None,
    )
    var_dict = {'l1': np.array([3.]),
                'epochs': np.array([4.]),
                'batch_size': np.array([4.]),
                'act_fn': np.array([2.]),
                'optimizer': np.array([11.]),
                'dropout_prob': np.array([0.01]),
                'lr_mult': np.array([1.]),
                'patience': np.array([2.]),
                'batch_norm': np.array([0.]),
                'initialization': np.array([0.])}
    g = generate_one_config_from_var_dict(var_dict=var_dict, fun_control=fun_control)
    # Since g is an iterator, we need to call next to get the values
    values = next(g)
    
    # Replace values['act_fn'] with the class name, i.e. .__class__.__name__ to get the class name
    values['act_fn'] = values['act_fn'].__class__.__name__
    # This is necessary for this comparison, because the act_fn is a class object and not a string
    
    expected = {'act_fn': 'ReLU',
                'batch_norm': False,
                'batch_size': 16,
                'dropout_prob': 0.01,
                'epochs': 16,
                'initialization': 'Default',
                'l1': 8,
                'lr_mult': 1.0,
                'optimizer': 'SGD',
                'patience': 4}

    assert values == expected