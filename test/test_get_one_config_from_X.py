import numpy as np
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.hyperparameters.values import get_core_model_from_name, add_core_model_to_fun_control, get_one_config_from_X
from torch.nn import ReLU

def test_get_dict_with_levels_and_types():
    core_model_name="light.regression.NNLinearRegressor"
    hyperdict=LightHyperDict
    fun_control = {}
    coremodel, core_model_instance = get_core_model_from_name(core_model_name)
    add_core_model_to_fun_control(
        core_model=core_model_instance,
        fun_control=fun_control,
        hyper_dict=hyperdict,
        filename=None,
    )
    X = np.array([[3.0e+00, 4.0e+00, 4.0e+00, 2.0e+00, 1.1e+01, 1.0e-02, 1.0e+00, 2.0e+00, 0.0e+00,
    0.0e+00]])
    print(X)
    values = get_one_config_from_X(X, fun_control)
    
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