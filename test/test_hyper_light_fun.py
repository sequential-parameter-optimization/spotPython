import pytest
from spotPython.utils.init import fun_control_init
from spotPython.light.regression.netlightregression import NetLightRegression
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.hyperparameters.values import add_core_model_to_fun_control, get_default_hyperparameters_as_array
from spotPython.fun.hyperlight import HyperLight
from spotPython.data.diabetes import Diabetes
from spotPython.hyperparameters.values import set_control_key_value
import numpy as np


def test_hyper_light_fun():
    fun_control = fun_control_init(
        _L_in=10,
        _L_out=1,)

    dataset = Diabetes()
    set_control_key_value(control_dict=fun_control,
                          key="data_set",
                          value=dataset)


    add_core_model_to_fun_control(core_model=NetLightRegression,
                                fun_control=fun_control,
                                hyper_dict=LightHyperDict)
    hyper_light = HyperLight(seed=126, log_level=50)
    X = get_default_hyperparameters_as_array(fun_control)
    # combine X and X to a np.array with shape (2, n_hyperparams)
    # so that two values are returned
    X = np.vstack((X, X))
    y = hyper_light.fun(X, fun_control)
    assert y.shape == (2,)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
