from spotpython.hyperparameters.values import get_var_name


def test_get_var_name():
    fun_control = {
        "core_model_hyper_dict": {
            "leaf_prediction": {
                "levels": ["mean", "model", "adaptive"],
                "type": "factor",
                "default": "mean",
                "core_model_parameter_type": "str",
            },
            "leaf_model": {
                "levels": ["linear_model.LinearRegression", "linear_model.PARegressor", "linear_model.Perceptron"],
                "type": "factor",
                "default": "LinearRegression",
                "core_model_parameter_type": "instance",
            },
            "splitter": {
                "levels": ["EBSTSplitter", "TEBSTSplitter", "QOSplitter"],
                "type": "factor",
                "default": "EBSTSplitter",
                "core_model_parameter_type": "instance()",
            },
            "binary_split": {"levels": [0, 1], "type": "factor", "default": 0, "core_model_parameter_type": "bool"},
            "stop_mem_management": {
                "levels": [0, 1],
                "type": "factor",
                "default": 0,
                "core_model_parameter_type": "bool",
            },
        }
    }
    # fun_control has 5 keys (hyperparameters)
    assert len(get_var_name(fun_control)) == 5
