import pytest
from spotPython.hyperparameters.values import set_self_attribute

class DummyClass:
    pass

def test_set_self_attribute():
    dummy = DummyClass()

    # Test when key is not in dict
    set_self_attribute(dummy, "attribute1", "value1", {})
    assert dummy.attribute1 == "value1"

    # Test when key is in dict
    set_self_attribute(dummy, "attribute2", "value2", {"attribute2": "new_value2"})
    assert dummy.attribute2 == "new_value2"