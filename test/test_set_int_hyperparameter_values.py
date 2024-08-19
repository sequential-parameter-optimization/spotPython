from spotriver.hyperdict.river_hyper_dict import RiverHyperDict
from spotpython.utils.init import fun_control_init
from spotpython.hyperparameters.values import (
    set_int_hyperparameter_values,
    set_float_hyperparameter_values,
    set_boolean_hyperparameter_values,
    set_factor_hyperparameter_values,
)


def test_set_int_hyperparameter_values():
    # Initialize fun_control with expected structure and mock data
    fun_control = fun_control_init(core_model_name="forest.AMFRegressor", hyperdict=RiverHyperDict)

    # Apply modifications
    set_int_hyperparameter_values(fun_control, "n_estimators", 2, 5)

    # Retrieve updated hyperparameters
    updated_hyperparameters = fun_control["core_model_hyper_dict"]

    # Assertions to confirm hyperparameter modifications
    assert (
        updated_hyperparameters["n_estimators"]["lower"] == 2
    ), "Lower bound of 'n_estimators' was not correctly modified."
    assert (
        updated_hyperparameters["n_estimators"]["upper"] == 5
    ), "Upper bound of 'n_estimators' was not correctly modified."


def test_set_float_hyperparameter_values():
    # Initialize fun_control with the expected structure for this test
    fun_control = fun_control_init(core_model_name="forest.AMFRegressor", hyperdict=RiverHyperDict)

    # Apply float modifications
    set_float_hyperparameter_values(fun_control, "step", 0.2, 5.0)

    # Access updated hyperparameters
    updated_hyperparameters = fun_control["core_model_hyper_dict"]
    # Assertions to confirm float hyperparameter modifications
    assert updated_hyperparameters["step"]["lower"] == 0.2, "Lower bound of 'step' was not correctly modified."
    assert updated_hyperparameters["step"]["upper"] == 5.0, "Upper bound of 'step' was not correctly modified."


def test_set_boolean_hyperparameter_values():
    # Initialize fun_control with the expected structure for this test
    fun_control = fun_control_init(core_model_name="forest.AMFRegressor", hyperdict=RiverHyperDict)

    set_boolean_hyperparameter_values(fun_control, "use_aggregation", 0, 0)

    # Access updated hyperparameters
    updated_hyperparameters = fun_control["core_model_hyper_dict"]
    # Assertions to confirm float hyperparameter modifications
    assert updated_hyperparameters["use_aggregation"]["lower"] == 0, "Lower bound of 'step' was not correctly modified."
    assert updated_hyperparameters["use_aggregation"]["upper"] == 0, "Upper bound of 'step' was not correctly modified."


def test_set_factor_hyperparameter_values():
    # Initialize fun_control with the expected structure for this test
    fun_control = fun_control_init(
        core_model_name="tree.HoeffdingTreeRegressor",
        hyperdict=RiverHyperDict,
    )

    set_factor_hyperparameter_values(fun_control, "leaf_model", ["LinearRegression", "Perceptron"])

    # Access updated hyperparameters
    updated_hyperparameters = fun_control["core_model_hyper_dict"]
    assert updated_hyperparameters["leaf_model"]["levels"] == [
        "LinearRegression",
        "Perceptron",
    ], "Levels of 'leaf_model' were not correctly modified."
    assert updated_hyperparameters["leaf_model"]["type"] == "factor", "Type of 'leaf_model' was not correctly modified."
    assert (
        updated_hyperparameters["leaf_model"]["default"] == "LinearRegression"
    ), "Default value of 'leaf_model' was not correctly modified."
    assert (
        updated_hyperparameters["leaf_model"]["transform"] == "None"
    ), "Transform of 'leaf_model' was not correctly modified."
    assert (
        updated_hyperparameters["leaf_model"]["class_name"] == "river.linear_model"
    ), "Class name of 'leaf_model' was not correctly modified."
    assert (
        updated_hyperparameters["leaf_model"]["core_model_parameter_type"] == "instance()"
    ), "Core model parameter type of 'leaf_model' was not correctly modified."
    assert (
        updated_hyperparameters["leaf_model"]["lower"] == 0
    ), "Lower bound of 'leaf_model' was not correctly modified."
    assert (
        updated_hyperparameters["leaf_model"]["upper"] == 1
    ), "Upper bound of 'leaf_model' was not correctly modified."
