import pytest
import river
import spotpython
from spotpython.hyperparameters.values import get_core_model_from_name


def test_valid_spotpython_model():
    model_name, model_instance = get_core_model_from_name("light.regression.NNLinearRegressor")
    assert model_name == "NNLinearRegressor"
    assert model_instance == spotpython.light.regression.NNLinearRegressor


def test_invalid_model_format():
    with pytest.raises(ValueError) as excinfo:
        get_core_model_from_name("invalidModelFormat")
    assert "Invalid core model name" in str(excinfo.value)


def test_nonexistent_model():
    with pytest.raises(ValueError) as excinfo:
        get_core_model_from_name("tree.NonExistentModel")
    assert "Model 'tree.NonExistentModel' not found" in str(excinfo.value)


def test_invalid_library_model():
    with pytest.raises(ValueError) as excinfo:
        get_core_model_from_name("invalidLibrary.SomeModel")
    assert "Model 'invalidLibrary.SomeModel' not found" in str(excinfo.value)
