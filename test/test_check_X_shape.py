import pytest
import numpy as np
from spotPython.utils.init import fun_control_init
from spotPython.light.regression.netlightregression import NetLightRegression
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.fun.hyperlight import HyperLight
from spotPython.hyperparameters.values import get_var_name


def test_check_X_shape():
    fun_control = fun_control_init()
    add_core_model_to_fun_control(core_model=NetLightRegression,
                                fun_control=fun_control,
                                hyper_dict=LightHyperDict)
    hyper_light = HyperLight(seed=126, log_level=50)
    n_hyperparams = len(get_var_name(fun_control))
    # generate a random np.array X with shape (2, n_hyperparams)
    X = np.random.rand(2, n_hyperparams)
    # assert that all values in X  and hyper_light.check_X_shape(X, fun_control) are equal
    assert np.all(X == hyper_light.check_X_shape(X, fun_control))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
