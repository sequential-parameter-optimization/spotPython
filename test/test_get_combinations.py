import pytest
from spotpython.utils.stats import get_combinations

def test_get_combinations_empty_list():
    """Test get_combinations with an empty list."""
    assert get_combinations([]) == []

def test_get_combinations_single_element():
    """Test get_combinations with a single element."""
    assert get_combinations([0]) == []

def test_get_combinations_two_elements():
    """Test get_combinations with two elements."""
    assert get_combinations([0, 1]) == [(0, 1)]

def test_get_combinations_multiple_elements():
    """Test get_combinations with multiple elements."""
    z_ind = [0, 1, 2, 3]
    expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert get_combinations(z_ind) == expected

def test_get_combinations_non_sequential_indices():
    """Test get_combinations with non-sequential indices."""
    z_ind = [10, 20, 30]
    expected = [(10, 20), (10, 30), (20, 30)]  # Indices are based on values, not indices
    assert get_combinations(z_ind) == expected