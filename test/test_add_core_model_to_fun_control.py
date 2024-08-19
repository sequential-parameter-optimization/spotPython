from spotpython.light.regression.netlightregression import NetLightRegression
from spotpython.utils.init import fun_control_init
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.hyperparameters.values import add_core_model_to_fun_control


def test_add_core_model_to_fun_control():
    fun_control = fun_control_init()
    add_core_model_to_fun_control(core_model=NetLightRegression, fun_control=fun_control, hyper_dict=LightHyperDict)
    assert fun_control["core_model"].__name__ == "NetLightRegression"
    assert isinstance(fun_control["core_model_hyper_dict"], dict)
