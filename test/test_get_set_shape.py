import pytest
import numpy as np
from spotpython.utils.convert import get_shape, set_shape

def test_get_shape():
    # Test 1D array
    x1 = np.array([1, 2, 3])
    assert get_shape(x1) == (3, None)

    # Test 2D array
    x2 = np.array([[1, 2], [3, 4], [5, 6]])
    assert get_shape(x2) == (3, 2)

    # Test empty 1D array
    x3 = np.array([])
    assert get_shape(x3) == (0, None)

    # Test empty 2D array
    x4 = np.empty((0, 2))
    assert get_shape(x4) == (0, 2)

    # Test invalid input (3D array)
    x5 = np.array([[[1, 2], [3, 4]]])
    with pytest.raises(ValueError, match="Input array must be 1D or 2D."):
        get_shape(x5)


def test_set_shape():
    # Test reshaping to 1D
    x1 = np.array([1, 2, 3, 4])
    result = set_shape(x1, (4, None))
    assert result.shape == (4,)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    # Test reshaping to 2D
    x2 = np.array([1, 2, 3, 4])
    result = set_shape(x2, (2, 2))
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))

    # Test reshaping from 2D to 1D
    x3 = np.array([[1, 2], [3, 4]])
    result = set_shape(x3, (4, None))
    assert result.shape == (4,)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    # Test invalid reshape (size mismatch)
    x4 = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Cannot reshape array of size 3 to shape \\(4,\\)"):
        set_shape(x4, (4, None))

    # Test invalid reshape (size mismatch for 2D)
    x5 = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="Cannot reshape array of size 4 to shape \\(3, 2\\)"):
        set_shape(x5, (3, 2))