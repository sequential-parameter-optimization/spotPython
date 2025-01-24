import pytest
import numpy as np
from spotpython.build.kriging import Kriging

def test_extract_from_bounds_example_1():
    # Example 1 from the docstring
    num_theta = 2
    num_p = 3
    S = Kriging(
        seed=124,
        n_theta=num_theta,
        n_p=num_p,
        optim_p=True,
        noise=True
    )
    bounds_array = np.array([1, 2, 3, 4, 5, 6])
    S._extract_from_bounds(new_theta_p_Lambda=bounds_array)
    assert np.array_equal(S.theta, [1, 2]), f"Expected theta to be [1, 2] but got {S.theta}"
    assert np.array_equal(S.p, [3, 4, 5]), f"Expected p to be [3, 4, 5] but got {S.p}"
    assert S.Lambda == 6, f"Expected Lambda to be 6 but got {S.Lambda}"

def test_extract_from_bounds_example_2():
    # Example 2 from the docstring
    num_theta = 1
    num_p = 1
    S = Kriging(
        seed=124,
        n_theta=num_theta,
        n_p=num_p,
        optim_p=False,
        noise=False
    )
    bounds_array = np.array([1])
    S._extract_from_bounds(new_theta_p_Lambda=bounds_array)
    assert np.array_equal(S.theta, [1]), f"Expected theta to be [1] but got {S.theta}"

def test_extract_from_bounds_example_3():
    # Example 3 from the docstring
    num_theta = 1
    num_p = 2
    S = Kriging(
        seed=124,
        n_theta=num_theta,
        n_p=num_p,
        optim_p=True,
        noise=True
    )
    bounds_array = np.array([1, 2, 3, 4])
    S._extract_from_bounds(new_theta_p_Lambda=bounds_array)
    assert np.array_equal(S.theta, [1]), f"Expected theta to be [1] but got {S.theta}"
    assert np.array_equal(S.p, [2, 3]), f"Expected p to be [2, 3] but got {S.p}"
    assert S.Lambda == 4, f"Expected Lambda to be 4 but got {S.Lambda}"

def test_extract_from_bounds_invalid_length():
    # Test with invalid length of new_theta_p_Lambda
    num_theta = 2
    num_p = 3
    S = Kriging(
        seed=124,
        n_theta=num_theta,
        n_p=num_p,
        optim_p=True,
        noise=True
    )
    bounds_array = np.array([1, 2, 3])  # Invalid length
    with pytest.raises(ValueError, match="Input array must have at least 6 elements."):
        S._extract_from_bounds(new_theta_p_Lambda=bounds_array)

if __name__ == "__main__":
    pytest.main()