from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.hyperparameters.values import get_dict_with_levels_and_types, get_core_model_from_name, add_core_model_to_fun_control
from torch.nn import ReLU

def test_get_dict_with_levels_and_types():
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
    v = {'act_fn': 2,
                'batch_norm': 0,
                'batch_size': 4,
                'dropout_prob': 0.01,
                'epochs': 4,
                'initialization': 0,
                'l1': 3,
                'lr_mult': 1.0,
                'optimizer': 11,
                'patience': 2}

    values = get_dict_with_levels_and_types(fun_control=fun_control, v=v)
    
    # Replace values['act_fn'] with the class name, i.e. .__class__.__name__ to get the class name
    values['act_fn'] = values['act_fn'].__class__.__name__
    # This is necessary for this comparison, because the act_fn is a class object and not a string
       
        
    expected = {'act_fn': 'ReLU',
                    'batch_norm': False,
                    'batch_size': 4,
                    'dropout_prob': 0.01,
                    'epochs': 4,
                    'initialization': 'Default',
                    'l1': 3,
                    'lr_mult': 1.0,
                    'optimizer': 'SGD',
                    'patience': 2}

    assert values == expected