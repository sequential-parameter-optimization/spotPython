import pytest
import numpy as np
from spotPython.utils.init import fun_control_init
from spotPython.hyperparameters.values import set_fun_control_fun_evals, get_fun_control_fun_evals
from spotPython.hyperparameters.values import get_bound_values
from spotPython.hyperparameters.values import set_fun_control_fun_repeats, get_fun_control_fun_repeats
from spotPython.hyperparameters.values import set_fun_control_seed, get_fun_control_seed
from spotPython.hyperparameters.values import set_fun_control_sigma, get_fun_control_sigma
from spotPython.hyperparameters.values import set_fun_control_max_time, get_fun_control_max_time
from spotPython.hyperparameters.values import get_fun_control_key_value, set_fun_control_key_value



def setup_function():
    global fun_control
    fun_control = fun_control_init()

def test_set_fun_control_fun_evals():
    set_fun_control_fun_evals(fun_control, 5)
    assert fun_control["fun_evals"] == 5

def test_get_fun_control_fun_evals():
    set_fun_control_fun_evals(fun_control, 10)
    result = get_fun_control_fun_evals(fun_control)
    assert result == 10

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

def test_set_fun_control_fun_repeats():
    fun_control = fun_control_init()
    set_fun_control_fun_repeats(fun_control, 5)
    assert fun_control["fun_repeats"] == 5

def test_get_fun_control_fun_repeats():
    # Test when "fun_repeats" is not in fun_control
    fun_control = None
    assert get_fun_control_fun_repeats(fun_control) is None

    # Test when "fun_repeats" is in fun_control
    fun_control = fun_control_init()
    set_fun_control_fun_repeats(fun_control, 10)
    assert get_fun_control_fun_repeats(fun_control) == 10


def test_set_fun_control_seed():
    fun_control = fun_control_init()
    set_fun_control_seed(fun_control, 5)
    assert fun_control["seed"] == 5

def test_get_fun_control_seed():
    # Test when "seed" is not in fun_control
    fun_control = None
    assert get_fun_control_seed(fun_control) is None

    # Test when "seed" is in fun_control
    fun_control = fun_control_init()
    set_fun_control_seed(fun_control, 10)
    assert get_fun_control_seed(fun_control) == 10

def test_set_fun_control_sigma():
    fun_control = fun_control_init()
    set_fun_control_sigma(fun_control, 5.0)
    assert fun_control["sigma"] == 5.0

def test_get_fun_control_sigma():
    
    # Test when "sigma" is not in fun_control
    fun_control = None
    assert get_fun_control_sigma(fun_control) is None

    # Test when "sigma" is in fun_control
    fun_control = fun_control_init()
    set_fun_control_sigma(fun_control, 10.0)
    assert get_fun_control_sigma(fun_control) == 10.0


def test_set_fun_control_max_time():
    fun_control = fun_control_init()
    set_fun_control_max_time(fun_control, 5.0)
    assert fun_control["max_time"] == 5.0

def test_get_fun_control_max_time():
    fun_control = fun_control_init()

    # Test when "max_time" is not in fun_control
    assert get_fun_control_max_time(fun_control) is None

    # Test when fun_control is None
    assert get_fun_control_max_time() is None

    # Test when "max_time" is in fun_control
    set_fun_control_max_time(fun_control, 10.0)
    assert get_fun_control_max_time(fun_control) == 10.0

def test_set_fun_control_key_value():
    fun_control = fun_control_init()

    # Test when key is not in fun_control and replace is False
    set_fun_control_key_value(fun_control, "key1", "value1")
    assert fun_control["key1"] == "value1"

    # Test when key is in fun_control and replace is False
    set_fun_control_key_value(fun_control, "key1", "value2")
    assert fun_control["key1"] == "value1"

    # Test when key is in fun_control and replace is True
    set_fun_control_key_value(fun_control, "key1", "value2", replace=True)
    assert fun_control["key1"] == "value2"

    # Test when key is not in fun_control and replace is True
    set_fun_control_key_value(fun_control, "key2", "value3", replace=True)
    assert fun_control["key2"] == "value3"

def test_get_fun_control_key_value():
    fun_control = fun_control_init()

    # Test when key is not in fun_control
    assert get_fun_control_key_value(fun_control, "key1") is None

    # Test when fun_control is None
    assert get_fun_control_key_value() is None

    # Test when key is in fun_control
    set_fun_control_key_value(fun_control, "key1", "value1")
    assert get_fun_control_key_value(fun_control, "key1") == "value1"
    