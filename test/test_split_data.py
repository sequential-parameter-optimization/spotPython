import pytest
from spotpython.utils.split import calculate_data_split

def test_calculate_data_split_proportion():
    # Test with proportion for test size
    result = calculate_data_split(0.2, 1000)
    assert result == (0.8, 0.16, 0.64, 0.2), f"Unexpected result: {result}"

def test_calculate_data_split_absolute():
    # Test with absolute number for test size
    result = calculate_data_split(200, 1000)
    assert result == (800, 160, 640, 200), f"Unexpected result: {result}"

def test_calculate_data_split_invalid():
    # Test with invalid input where test size exceeds full size
    with pytest.raises(ValueError):
        calculate_data_split(1200, 1000)