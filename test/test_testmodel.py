from spotPython.utils.init import fun_control_init
from spotPython.light.regression.netlightregression import NetLightRegression
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.hyperparameters.values import add_core_model_to_fun_control, get_default_hyperparameters_as_array
from spotPython.data.diabetes import Diabetes
from spotPython.hyperparameters.values import set_control_key_value
from spotPython.hyperparameters.values import get_var_name, assign_values, generate_one_config_from_var_dict
import spotPython.light.testmodel as tm


def test_testmodel():
    fun_control = fun_control_init(_L_in=10, _L_out=1, _torchmetric="mean_squared_error")

    dataset = Diabetes()
    set_control_key_value(control_dict=fun_control, key="data_set", value=dataset, replace=True)

    add_core_model_to_fun_control(core_model=NetLightRegression, fun_control=fun_control, hyper_dict=LightHyperDict)
    X = get_default_hyperparameters_as_array(fun_control)
    var_dict = assign_values(X, get_var_name(fun_control))
    for config in generate_one_config_from_var_dict(var_dict, fun_control):
        print(config)
        y_test = tm.test_model(config, fun_control)
        break
    # check if y is a float
    assert isinstance(y_test[0], float)
