import pytest
from spotpython.hyperparameters.values import set_hyperparameter


def test_set_hyperparameter_int():
    fun_control = {"core_model_hyper_dict": {"n_estimators": {"type": "int", "default": 10, "lower": 2, "upper": 1000}}}
    set_hyperparameter(fun_control, "n_estimators", [2, 5])
    assert fun_control["core_model_hyper_dict"]["n_estimators"]["lower"] == 2
    assert fun_control["core_model_hyper_dict"]["n_estimators"]["upper"] == 5


def test_set_hyperparameter_float():
    fun_control = {"core_model_hyper_dict": {"step": {"type": "float", "default": 1.0, "lower": 0.1, "upper": 10.0}}}
    set_hyperparameter(fun_control, "step", [0.2, 5.0])
    assert fun_control["core_model_hyper_dict"]["step"]["lower"] == 0.2
    assert fun_control["core_model_hyper_dict"]["step"]["upper"] == 5.0


def test_set_hyperparameter_boolean():
    fun_control = {
        "core_model_hyper_dict": {
            "use_aggregation": {"type": "boolean", "default": 1, "lower": 0, "upper": 1, "levels": [0, 1]}
        }
    }
    set_hyperparameter(fun_control, "use_aggregation", [False, True])
    assert fun_control["core_model_hyper_dict"]["use_aggregation"]["lower"] is False
    assert fun_control["core_model_hyper_dict"]["use_aggregation"]["upper"] is True


def test_set_hyperparameter_factor():
    fun_control = {
        "core_model_hyper_dict": {"leaf_model": {"type": "factor", "default": "LinearRegression", "upper": 2}}
    }
    set_hyperparameter(fun_control, "leaf_model", ["LinearRegression", "Perceptron"])
    assert fun_control["core_model_hyper_dict"]["leaf_model"]["levels"] == ["LinearRegression", "Perceptron"]
    assert fun_control["core_model_hyper_dict"]["leaf_model"]["upper"] == 1


def test_set_hyperparameter_single_string():
    fun_control = {
        "core_model_hyper_dict": {"leaf_model": {"type": "factor", "default": "LinearRegression", "upper": 0}}
    }
    set_hyperparameter(fun_control, "leaf_model", "LinearRegression")
    assert fun_control["core_model_hyper_dict"]["leaf_model"]["levels"] == ["LinearRegression"]
    assert fun_control["core_model_hyper_dict"]["leaf_model"]["upper"] == 0


def test_set_hyperparameter_invalid_type():
    fun_control = {"core_model_hyper_dict": {"n_estimators": {"type": "int", "default": 10, "lower": 2, "upper": 1000}}}
    with pytest.raises(ValueError):
        set_hyperparameter(fun_control, "n_estimators", [2, "five"])


def test_set_hyperparameter_invalid_values_type():
    fun_control = {"core_model_hyper_dict": {"n_estimators": {"type": "int", "default": 10, "lower": 2, "upper": 1000}}}
    with pytest.raises(TypeError):
        set_hyperparameter(fun_control, "n_estimators", 2, 5)
