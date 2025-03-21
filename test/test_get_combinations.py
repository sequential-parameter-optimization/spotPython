import pytest
from spotpython.utils.stats import get_combinations

def test_get_combinations_empty_list():
    """Test get_combinations with an empty list."""
    assert get_combinations([]) == []
    assert get_combinations([], type="indices") == []
    assert get_combinations([], type="values") == []

def test_get_combinations_single_element():
    """Test get_combinations with a single element."""
    assert get_combinations([0]) == []
    assert get_combinations([0], type="indices") == []
    assert get_combinations([0], type="values") == []

def test_get_combinations_two_elements():
    """Test get_combinations with two elements."""
    assert get_combinations([0, 1]) == [(0, 1)]
    assert get_combinations([0, 1], type="indices") == [(0, 1)]
    assert get_combinations([0, 1], type="values") == [(0, 1)]

def test_get_combinations_multiple_elements():
    """Test get_combinations with multiple elements."""
    z_ind = [0, 1, 2, 3]
    expected_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    expected_values = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert get_combinations(z_ind) == expected_indices
    assert get_combinations(z_ind, type="indices") == expected_indices
    assert get_combinations(z_ind, type="values") == expected_values

def test_get_combinations_non_sequential_indices():
    """Test get_combinations with non-sequential indices."""
    z_ind = [10, 20, 30]
    expected_indices = [(0, 1), (0, 2), (1, 2)]
    expected_values = [(10, 20), (10, 30), (20, 30)]
    assert get_combinations(z_ind) == expected_indices
    assert get_combinations(z_ind, type="indices") == expected_indices
    assert get_combinations(z_ind, type="values") == expected_values

def test_get_combinations_mixed_values():
    """Test get_combinations with mixed values."""
    z_ind = [0, 10, 20, 30]
    expected_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    expected_values = [(0, 10), (0, 20), (0, 30), (10, 20), (10, 30), (20, 30)]
    assert get_combinations(z_ind) == expected_indices
    assert get_combinations(z_ind, type="indices") == expected_indices
    assert get_combinations(z_ind, type="values") == expected_values

def test_get_combinations_invalid_type():
    """Test get_combinations with an invalid type."""
    with pytest.raises(ValueError):
        get_combinations([1, 2, 3], type="invalid")