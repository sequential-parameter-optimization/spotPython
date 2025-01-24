import pytest
from spotpython.build.kriging import Kriging

def test_set_de_bounds_default():
    # Create a Kriging instance with default settings
    S = Kriging()

    # Set de_bounds
    S._set_de_bounds()

    # Check if de_bounds is set correctly
    expected_de_bounds = [[S.min_theta, S.max_theta] for _ in range(S.n_theta)]
    assert S.de_bounds == expected_de_bounds

def test_set_de_bounds_with_optim_p():
    # Create a Kriging instance with optim_p=True
    S = Kriging(n_theta=2, n_p=2, optim_p=True)

    # Set de_bounds
    S._set_de_bounds()

    # Check if de_bounds is set correctly
    expected_de_bounds = [[S.min_theta, S.max_theta] for _ in range(S.n_theta)]
    expected_de_bounds += [[S.min_p, S.max_p] for _ in range(S.n_p)]
    assert S.de_bounds == expected_de_bounds

def test_set_de_bounds_with_noise():
    # Create a Kriging instance with noise=True
    S = Kriging(n_theta=2, n_p=2, noise=True)

    # Set de_bounds
    S._set_de_bounds()

    # Check if de_bounds is set correctly
    expected_de_bounds = [[S.min_theta, S.max_theta] for _ in range(S.n_theta)]
    expected_de_bounds.append([S.min_Lambda, S.max_Lambda])
    assert S.de_bounds == expected_de_bounds

def test_set_de_bounds_with_optim_p_and_noise():
    # Create a Kriging instance with optim_p=True and noise=True
    S = Kriging(n_theta=2, n_p=2, optim_p=True, noise=True)

    # Set de_bounds
    S._set_de_bounds()

    # Check if de_bounds is set correctly
    expected_de_bounds = [[S.min_theta, S.max_theta] for _ in range(S.n_theta)]
    expected_de_bounds += [[S.min_p, S.max_p] for _ in range(S.n_p)]
    expected_de_bounds.append([S.min_Lambda, S.max_Lambda])
    assert S.de_bounds == expected_de_bounds

def test_set_de_bounds_example_1():
    # Example 1 from the docstring
    S = Kriging()
    S._set_de_bounds()
    assert S.de_bounds == [[-3.0, 2.0]]

def test_set_de_bounds_example_2():
    # Example 2 from the docstring
    S = Kriging(n_theta=2, n_p=2, optim_p=True)
    S._set_de_bounds()
    assert S.de_bounds == [[-3.0, 2.0], [-3.0, 2.0], [1, 2], [1, 2]]

def test_set_de_bounds_example_3():
    # Example 3 from the docstring
    S = Kriging(n_theta=2, n_p=2, optim_p=True, noise=True)
    S._set_de_bounds()
    assert S.de_bounds == [[-3.0, 2.0], [-3.0, 2.0], [1, 2], [1, 2], [1e-09, 1.0]]

def test_set_de_bounds_example_4():
    # Example 4 from the docstring
    S = Kriging(n_theta=2, n_p=2, noise=True)
    S._set_de_bounds()
    assert S.de_bounds == [[-3.0, 2.0], [-3.0, 2.0], [1e-09, 1.0]]

if __name__ == "__main__":
    pytest.main()