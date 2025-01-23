def test_aggregate_mean_var_var():
    """
    Test aggregate mean and var.
    Combine a (6,3) array (X):
    [[1. 1. 1.]
    [1. 1. 1.]
    [2. 2. 2.]
    [2. 2. 2.]
    [1. 1. 1.]
    [1. 1. 1.]]
    and an (1,) array (y):
    [3. 3. 6. 6. 6. 6.]
    Expected results:
    array([[1., 1., 1.],
            [2., 2., 2.]]),
    array([4.5, 6. ]),
    array([3., 0.]).

    """
    import numpy as np
    from spotpython.utils.aggregate import aggregate_mean_var

    X_1 = np.ones((2, 3))
    y_1 = np.sum(X_1, axis=1)
    y_2 = 2 * y_1
    X_2 = np.append(X_1, 2 * X_1, axis=0)
    X = np.append(X_2, X_1, axis=0)
    y = np.append(y_1, y_2, axis=0)
    y = np.append(y, y_2, axis=0)
    Z = aggregate_mean_var(X, y, var_empirical=True)
    assert (Z[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).all()
    assert (Z[1] == np.array([4.5, 6.0])).all()
    assert (Z[2] == np.array([3.0, 0.0])).all()
