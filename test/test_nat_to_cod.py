import numpy as np
from spotpython.utils.transform import cod_to_nat_y, cod_to_nat_X


def test_cod_to_nat_y():
    # Test with cod_type = "norm"
    cod_y = np.array([0.5, 0.5, 0.5])
    min_y = np.array([0, 0, 0])
    max_y = np.array([1, 1, 1])
    y = cod_to_nat_y(cod_y, "norm", min_y, max_y)
    assert np.allclose(y, np.array([0.5, 0.5, 0.5]))

    # Test with cod_type = "std"
    mean_y = np.array([0, 0, 0])
    std_y = np.array([1, 1, 1])
    y = cod_to_nat_y(cod_y, "std", mean_y=mean_y, std_y=std_y)
    assert np.allclose(y, np.array([0.5, 0.5, 0.5]))

    # Test with cod_type = "other"
    y = cod_to_nat_y(cod_y, "other")
    assert np.allclose(y, np.array([0.5, 0.5, 0.5]))

    # Test with max_y - min_y = 0 for cod_type = "norm"
    min_y = np.array([0, 0, 0])
    max_y = np.array([0, 0, 0])
    y = cod_to_nat_y(cod_y, "norm", min_y, max_y)
    assert np.allclose(y, np.array([0, 0, 0]))

    # Test with max_y - min_y = 0 for cod_type = "std"
    mean_y = np.array([0, 0, 0])
    std_y = np.array([0, 0, 0])
    y = cod_to_nat_y(cod_y, "std", mean_y=mean_y, std_y=std_y)
    assert np.allclose(y, np.array([0, 0, 0]))


def test_cod_to_nat_X():
    # Test with cod_type = "norm"
    cod_X = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    min_X = np.array([0, 0, 0])
    max_X = np.array([1, 1, 1])
    X = cod_to_nat_X(cod_X, "norm", min_X, max_X)
    assert np.allclose(X, np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))

    # Test with cod_type = "std"
    mean_X = np.array([0, 0, 0])
    std_X = np.array([1, 1, 1])
    X = cod_to_nat_X(cod_X, "std", mean_X=mean_X, std_X=std_X)
    assert np.allclose(X, np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))

    # Test with cod_type = "other"
    X = cod_to_nat_X(cod_X, "other")
    assert np.allclose(X, np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))

    # Test with max_X - min_X = 0 for cod_type = "norm"
    min_X = np.array([0, 0, 0])
    max_X = np.array([0, 0, 0])
    X = cod_to_nat_X(cod_X, "norm", min_X, max_X)
    assert np.allclose(X, np.array([[0, 0, 0], [0, 0, 0]]))

    # Test with max_X - min_X = 0 for cod_type = "std"
    mean_X = np.array([0, 0, 0])
    std_X = np.array([0, 0, 0])
    X = cod_to_nat_X(cod_X, "std", mean_X=mean_X, std_X=std_X)
    assert np.allclose(X, np.array([[0, 0, 0], [0, 0, 0]]))
