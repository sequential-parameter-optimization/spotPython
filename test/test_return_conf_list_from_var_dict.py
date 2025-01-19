from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.hyperparameters.values import get_core_model_from_name, add_core_model_to_fun_control, return_conf_list_from_var_dict
import numpy as np


def test_return_conf_list_from_var_dict():
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
    var_dict = {'l1': np.array([3., 4.]),
                'epochs': np.array([4., 3.]),
                'batch_size': np.array([4., 4.]),
                'act_fn': np.array([2., 1.]),
                'optimizer': np.array([11., 10.]),
                'dropout_prob': np.array([0.01, 0.]),
                'lr_mult': np.array([1., 1.1]),
                'patience': np.array([2., 3.]),
                'batch_norm': np.array([0., 1.]),
                'initialization': np.array([0., 1.])}
    cl = return_conf_list_from_var_dict(var_dict=var_dict, fun_control=fun_control)
    
    values = cl[0]
    
    # Replace values['act_fn'] with the class name, i.e. .__class__.__name__ to get the class name
    values['act_fn'] = values['act_fn'].__class__.__name__
    # This is necessary for this comparison, because the act_fn is a class object and not a string
    
    expected = {'l1': 8,
                'epochs': 16,
                'batch_size': 16,
                'act_fn': 'ReLU',
                'optimizer': 'SGD',
                'dropout_prob': 0.01,
                'lr_mult': 1.0,
                'patience': 4,
                'batch_norm': False,
                'initialization': 'Default'}
    
    assert values == expected
    
    values = cl[1]
    
    # Replace values['act_fn'] with the class name, i.e. .__class__.__name__ to get the class name
    values['act_fn'] = values['act_fn'].__class__.__name__
    # This is necessary for this comparison, because the act_fn is a class object and not a string
    
    expected = {'l1': 16,
                'epochs': 8,
                'batch_size': 16,
                'act_fn': 'Tanh',
                'optimizer': 'Rprop',
                'dropout_prob': 0.0,
                'lr_mult': 1.1,
                'patience': 8,
                'batch_norm': True,
                'initialization': 'kaiming_uniform'}

    assert values == expected