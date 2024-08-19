import pytest
from spotpython.utils.init import get_feature_names  # Replace 'your_module_name' with the actual module name


class MockDataSet:
    def __init__(self, names):
        self.names = names


def test_get_feature_names_success():
    fun_control = {"data_set": MockDataSet(names=["feature1", "feature2", "feature3"])}
    feature_names = get_feature_names(fun_control)
    assert feature_names == ["feature1", "feature2", "feature3"]


def test_get_feature_names_missing_data_set_key():
    fun_control = {}
    with pytest.raises(ValueError, match="'data_set' key not found or is None in 'fun_control'"):
        get_feature_names(fun_control)


def test_get_feature_names_data_set_none():
    fun_control = {"data_set": None}
    with pytest.raises(ValueError, match="'data_set' key not found or is None in 'fun_control'"):
        get_feature_names(fun_control)


def test_get_feature_names_empty_names():
    fun_control = {"data_set": MockDataSet(names=[])}
    feature_names = get_feature_names(fun_control)
    assert feature_names == []
