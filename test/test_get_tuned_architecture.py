from spotpython.data.diabetes import Diabetes
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.fun.hyperlight import HyperLight
from spotpython.utils.init import fun_control_init, design_control_init
from spotpython.spot import Spot
import numpy as np
from spotpython.hyperparameters.values import set_hyperparameter, get_tuned_architecture


def test_get_tuned_architecture():
    fun_control = fun_control_init(
        force_run=True,
        PREFIX="get_one_config_from_X",
        fun_evals=5,
        max_time=1,
        data_set = Diabetes(),
        core_model_name="light.regression.NNLinearRegressor",
        hyperdict=LightHyperDict,
        _L_in=10,
        _L_out=1)

    set_hyperparameter(fun_control, "epochs", [2,2])
    set_hyperparameter(fun_control, "patience", [1,1])
    design_control = design_control_init(init_size=4)

    fun = HyperLight().fun
    S = Spot(fun=fun,fun_control=fun_control, design_control=design_control)
    S.run()
    conf = get_tuned_architecture(S)
    
    # Consider transformation of the values
    assert conf['epochs'] == 4