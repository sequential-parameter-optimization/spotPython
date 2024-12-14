import pytest
import numpy as np
from spotpython.utils.repair import apply_penalty_NA

def test_apply_penalty_NA():
    # Test with no NaN values
    y = np.array([1, 2, 3])
    penalty_NA = 0
    y_cleaned = apply_penalty_NA(y, penalty_NA)
    np.testing.assert_array_equal(y_cleaned, y)

    # Test with NaN values
    y = np.array([1, np.nan, 2])
    penalty_NA = 0
    y_cleaned = apply_penalty_NA(y, penalty_NA)
    assert y_cleaned[1] != np.nan
    assert y_cleaned[1] != 0  # Because of the random noise

    # Test with different penalty_NA
    penalty_NA = 5
    y_cleaned = apply_penalty_NA(y, penalty_NA)
    assert y_cleaned[1] != np.nan
    assert y_cleaned[1] != 5  # Because of the random noise

    # Test with stop_on_zero_return
    y = np.array([np.nan])
    with pytest.raises(ValueError):
        apply_penalty_NA(y, penalty_NA, stop_on_zero_return=True)

    # Test with invalid input types
    with pytest.raises(TypeError):
        apply_penalty_NA([1, 2, 3], penalty_NA)
    with pytest.raises(TypeError):
        apply_penalty_NA(y, penalty_NA, sd="0.1")
    with pytest.raises(TypeError):
        apply_penalty_NA(y, penalty_NA, stop_on_zero_return="False")