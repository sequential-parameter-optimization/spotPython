import pytest
import numpy as np
from spotpython.utils.init import fun_control_init
from spotpython.hyperparameters.values import get_bound_values
from spotpython.hyperparameters.values import get_control_key_value, set_control_key_value
from spotpython.hyperparameters.values import get_var_type_from_var_name
from spotpython.hyperparameters.values import add_core_model_to_fun_control
from spotpython.light.regression.netlightregression import NetLightRegression
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.utils.device import getDevice
from spotpython.data.diabetes import Diabetes
from spotpython.hyperparameters.values import get_ith_hyperparameter_name_from_fun_control
from spotpython.hyperparameters.values import set_control_hyperparameter_value


def test_get_bound_values():
    fun_control = {"core_model_hyper_dict": {"a": {"upper": 1, "lower": 0}, "b": {"upper": 2, "lower": -1}}}

    # Test with bound="upper" and as_list=True
    result = get_bound_values(fun_control, "upper", as_list=True)
    assert result == [1, 2]

    # Test with bound="lower" and as_list=True
    result = get_bound_values(fun_control, "lower", as_list=True)
    assert result == [0, -1]

    # Test with bound="upper" and as_list=False
    result = get_bound_values(fun_control, "upper", as_list=False)
    assert np.array_equal(result, np.array([1, 2]))

    # Test with bound="lower" and as_list=False
    result = get_bound_values(fun_control, "lower", as_list=False)
    assert np.array_equal(result, np.array([0, -1]))

    # Test with invalid bound
    with pytest.raises(ValueError):
        get_bound_values(fun_control, "invalid", as_list=True)


def test_set_control_key_value():
    fun_control = fun_control_init()

    # Test when key is not in fun_control and replace is False
    set_control_key_value(control_dict=fun_control, key="key1", value="value1")
    assert fun_control["key1"] == "value1"

    # Test when key is in fun_control and replace is False
    set_control_key_value(control_dict=fun_control, key="key1", value="value2")
    assert fun_control["key1"] == "value1"

    # Test when key is in fun_control and replace is True
    set_control_key_value(control_dict=fun_control, key="key1", value="value2", replace=True)
    assert fun_control["key1"] == "value2"

    # Test when key is not in fun_control and replace is True
    set_control_key_value(control_dict=fun_control, key="key2", value="value3", replace=True)
    assert fun_control["key2"] == "value3"


def test_get_control_key_value():
    fun_control = fun_control_init()

    # Test when key is not in fun_control
    assert get_control_key_value(control_dict=fun_control, key="key1") is None

    # Test when fun_control is None
    assert get_control_key_value() is None

    # Test when key is in fun_control
    set_control_key_value(control_dict=fun_control, key="key1", value="value1")
    assert get_control_key_value(control_dict=fun_control, key="key1") == "value1"


def test_get_var_type_from_var_name():
    fun_control = fun_control_init()
    add_core_model_to_fun_control(core_model=NetLightRegression, fun_control=fun_control, hyper_dict=LightHyperDict)
    var_type = get_control_key_value(control_dict=fun_control, key="var_type")
    var_name = get_control_key_value(control_dict=fun_control, key="var_name")
    vn = "l1"
    assert var_type[var_name.index(vn)] == "int"
    assert get_var_type_from_var_name(fun_control, vn) == "int"
    vn = "initialization"
    assert var_type[var_name.index(vn)] == "factor"
    assert get_var_type_from_var_name(fun_control, vn) == "factor"


def test_get_th_hyperparameter_name_from_fun_control():
    fun_control = fun_control_init(
        _L_in=10,
        _L_out=1,
        PREFIX="test_values_2",
        TENSORBOARD_CLEAN=True,
        device=getDevice(),
        enable_progress_bar=False,
        fun_evals=15,
        log_level=10,
        max_time=1,
        num_workers=0,
        show_progress=True,
        tolerance_x=np.sqrt(np.spacing(1)),
    )
    dataset = Diabetes()
    set_control_key_value(control_dict=fun_control, key="data_set", value=dataset, replace=True)
    add_core_model_to_fun_control(core_model=NetLightRegression, fun_control=fun_control, hyper_dict=LightHyperDict)

    set_control_hyperparameter_value(fun_control, "l1", [3, 8])
    set_control_hyperparameter_value(fun_control, "optimizer", ["Adam", "AdamW", "Adamax", "NAdam"])
    assert get_ith_hyperparameter_name_from_fun_control(fun_control, key="optimizer", i=0) == "Adam"
    def test_assign_values():
        # Test with a 2x2 array
        X = np.array([[1, 2], [3, 4]])
        var_list = ['a', 'b']
        result = assign_values(X, var_list)
        expected = {'a': np.array([1, 3]), 'b': np.array([2, 4])}
        assert result.keys() == expected.keys()
        for key in result:
            assert np.array_equal(result[key], expected[key])

        # Test with a 3x2 array
        X = np.array([[1, 2], [3, 4], [5, 6]])
        var_list = ['a', 'b']
        result = assign_values(X, var_list)
        expected = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}
        assert result.keys() == expected.keys()
        for key in result:
            assert np.array_equal(result[key], expected[key])

        # Test with a 3x3 array
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        var_list = ['a', 'b', 'c']
        result = assign_values(X, var_list)
        expected = {'a': np.array([1, 4, 7]), 'b': np.array([2, 5, 8]), 'c': np.array([3, 6, 9])}
        assert result.keys() == expected.keys()
        for key in result:
            assert np.array_equal(result[key], expected[key])

        # Test with empty array and empty var_list
        X = np.array([[]])
        var_list = []
        result = assign_values(X, var_list)
        expected = {}
        assert result == expected

        # Test with mismatched var_list length
        X = np.array([[1, 2], [3, 4]])
        var_list = ['a']
        with pytest.raises(IndexError):
            assign_values(X, var_list)

