import pytest
from spotpython.utils.transform import transform_hyper_parameter_values

def test_transform_hyper_parameter_values_int():
    fun_control = {
        "core_model_hyper_dict": {
            "max_depth": {
                "type": "int",
                "default": 20,
                "transform": "transform_power_2",
                "lower": 2,
                "upper": 20
            }
        }
    }
    hyper_parameter_values = {'max_depth': 2}
    expected = {'max_depth': 4}
    result = transform_hyper_parameter_values(fun_control, hyper_parameter_values)
    assert result == expected

def test_transform_hyper_parameter_values_factor():
    fun_control = {
        "core_model_hyper_dict": {
            "leaf_prediction": {
                "type": "factor",
                "transform": "None",
                "default": "mean",
                "levels": ["mean", "model", "adaptive"],
                "core_model_parameter_type": "str"
            }
        }
    }
    hyper_parameter_values = {'leaf_prediction': 'mean'}
    expected = {'leaf_prediction': 'mean'}
    result = transform_hyper_parameter_values(fun_control, hyper_parameter_values)
    assert result == expected

def test_transform_hyper_parameter_values_multiple():
    fun_control = {
        "core_model_hyper_dict": {
            "l1": {
                "type": "int",
                "default": 3,
                "transform": "transform_power_2_int",
                "lower": 3,
                "upper": 8
            },
            "epochs": {
                "type": "int",
                "default": 4,
                "transform": "transform_power_2_int",
                "lower": 4,
                "upper": 9
            },
            "batch_size": {
                "type": "int",
                "default": 4,
                "transform": "transform_power_2_int",
                "lower": 1,
                "upper": 4
            },
            "act_fn": {
                "levels": [
                    "Sigmoid",
                    "Tanh",
                    "ReLU",
                    "LeakyReLU",
                    "ELU",
                    "Swish"
                ],
                "type": "factor",
                "default": "ReLU",
                "transform": "None",
                "class_name": "spotpython.torch.activation",
                "core_model_parameter_type": "instance()",
                "lower": 0,
                "upper": 5
            },
            "optimizer": {
                "levels": [
                    "Adadelta",
                    "Adagrad",
                    "Adam",
                    "AdamW",
                    "SparseAdam",
                    "Adamax",
                    "ASGD",
                    "NAdam",
                    "RAdam",
                    "RMSprop",
                    "Rprop",
                    "SGD"
                ],
                "type": "factor",
                "default": "SGD",
                "transform": "None",
                "class_name": "torch.optim",
                "core_model_parameter_type": "str",
                "lower": 0,
                "upper": 11
            },
            "dropout_prob": {
                "type": "float",
                "default": 0.01,
                "transform": "None",
                "lower": 0.0,
                "upper": 0.25
            },
            "lr_mult": {
                "type": "float",
                "default": 1.0,
                "transform": "None",
                "lower": 0.1,
                "upper": 10.0
            },
            "patience": {
                "type": "int",
                "default": 2,
                "transform": "transform_power_2_int",
                "lower": 2,
                "upper": 6
            },
            "batch_norm": {
                "levels": [
                    0,
                    1
                ],
                "type": "factor",
                "default": 0,
                "transform": "None",
                "core_model_parameter_type": "bool",
                "lower": 0,
                "upper": 1
            },
            "initialization": {
                "levels": [
                    "Default",
                    "kaiming_uniform",
                    "kaiming_normal",
                    "xavier_uniform",
                    "xavier_normal"
                ],
                "type": "factor",
                "default": "Default",
                "transform": "None",
                "core_model_parameter_type": "str",
                "lower": 0,
                "upper": 4
            }
        }
    }
    hyper_parameter_values = {
        'l1': 2,
        'epochs': 3,
        'batch_size': 4,
        'act_fn': 'ReLU',
        'optimizer': 'SGD',
        'dropout_prob': 0.01,
        'lr_mult': 1.0,
        'patience': 3,
        'batch_norm': 0,
        'initialization': 'Default'
    }
    expected = {
        'l1': 4,
        'epochs': 8,
        'batch_size': 16,
        'act_fn': 'ReLU',
        'optimizer': 'SGD',
        'dropout_prob': 0.01,
        'lr_mult': 1.0,
        'patience': 8,
        'batch_norm': 0,
        'initialization': 'Default'
    }
    result = transform_hyper_parameter_values(fun_control, hyper_parameter_values)
    assert result == expected