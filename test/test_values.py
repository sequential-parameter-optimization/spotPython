import pytest
import numpy as np
from spotPython.utils.init import fun_control_init
from spotPython.hyperparameters.values import get_bound_values
from spotPython.hyperparameters.values import get_control_key_value, set_control_key_value


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
    set_control_key_value(fun_control, "key1", "value1")
    assert fun_control["key1"] == "value1"

    # Test when key is in fun_control and replace is False
    set_control_key_value(fun_control, "key1", "value2")
    assert fun_control["key1"] == "value1"

    # Test when key is in fun_control and replace is True
    set_control_key_value(fun_control, "key1", "value2", replace=True)
    assert fun_control["key1"] == "value2"

    # Test when key is not in fun_control and replace is True
    set_control_key_value(fun_control, "key2", "value3", replace=True)
    assert fun_control["key2"] == "value3"

def test_get_control_key_value():
    fun_control = fun_control_init()

    # Test when key is not in fun_control
    assert get_control_key_value(fun_control, "key1") is None

    # Test when fun_control is None
    assert get_control_key_value() is None

    # Test when key is in fun_control
    set_control_key_value(fun_control, "key1", "value1")
    assert get_control_key_value(fun_control, "key1") == "value1"
    