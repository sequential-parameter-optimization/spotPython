import pytest
from spotPython.utils.init import fun_control_init
from spotPython.light.regression.netlightregression import NetLightRegression
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.hyperparameters.values import add_core_model_to_fun_control, get_default_hyperparameters_as_array
from spotPython.data.diabetes import Diabetes
from spotPython.hyperparameters.values import set_data_set
from spotPython.hyperparameters.values import get_var_name, assign_values, generate_one_config_from_var_dict
from spotPython.light.trainmodel import train_model


def test_trainmodel():
    fun_control = fun_control_init(
        _L_in=10,
        _L_out=1,)

    dataset = Diabetes()
    set_data_set(fun_control=fun_control,
                    data_set=dataset)

    add_core_model_to_fun_control(core_model=NetLightRegression,
                                fun_control=fun_control,
                                hyper_dict=LightHyperDict)
    X = get_default_hyperparameters_as_array(fun_control)
    var_dict = assign_values(X, get_var_name(fun_control))
    for config in generate_one_config_from_var_dict(var_dict, fun_control):
        y_train = train_model(config, fun_control)
        break
    # check if y is a float
    assert isinstance(y_train, float)
    

if __name__ == "__main__":
    pytest.main(["-v", __file__])
