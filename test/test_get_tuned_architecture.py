from spotpython.data.diabetes import Diabetes
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.fun.hyperlight import HyperLight
from spotpython.utils.init import fun_control_init
from spotpython.spot import Spot
import numpy as np
from spotpython.hyperparameters.values import get_one_config_from_X


def test_get_tuned_architecture():
    fun_control = fun_control_init(
        force_run=False,
        PREFIX="get_one_config_from_X",
        save_experiment=True,
        fun_evals=10,
        max_time=1,
        data_set = Diabetes(),
        core_model_name="light.regression.NNLinearRegressor",
        hyperdict=LightHyperDict,
        _L_in=10,
        _L_out=1)
    X = np.array([[ 4., 2., 1., 3.,  11., 0.03480567, 7.98104224,  1., 1., 0.]])
    values = get_one_config_from_X(X, fun_control)
    
    # Replace values['act_fn'] with the class name, i.e. .__class__.__name__ to get the class name
    values['act_fn'] = values['act_fn'].__class__.__name__
    # This is necessary for this comparison, because the act_fn is a class object and not a string
    
    expected = {'l1': 16,
                'epochs': 4,
                'batch_size': 2,
                'act_fn': 'LeakyReLU',
                'optimizer': 'SGD',
                'dropout_prob': 0.03480567,
                'lr_mult': 7.98104224,
                'patience': 2,
                'batch_norm': True,
                'initialization': 'Default'}
    assert values == expected