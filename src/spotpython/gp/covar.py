import numpy as np
import pandas as pd


def prepare_X(X: np.ndarray) -> np.ndarray:
    # check if X is a dataframe and convert to numpy array
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    return X


def covar_sep(col, X1, n1, X2, n2, d, g) -> np.ndarray:
    """
    Calculate the correlation (K) between X1 and X2 with
    a separable power exponential correlation function
    with range d and nugget g.

    Args:
        col (int): Number of columns in the input matrices X1 and X2.
        X1 (ndarray): First input matrix of shape (n1, col).
        n1 (int): Number of rows in the first input matrix X1.
        X2 (ndarray): Second input matrix of shape (n2, col).
        n2 (int): Number of rows in the second input matrix X2.
        d (ndarray): Array of length col representing the range parameters.
        g (float): Nugget parameter.

    Returns:
        ndarray: The calculated covariance matrix K of shape (n1, n2).

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.covar import covar_sep
        >>> col = 2
        >>> X1 = np.array([[1, 2], [3, 4], [5, 6]])
        >>> n1 = 3
        >>> X2 = np.array([[7, 8], [9, 10]])
        >>> n2 = 2
        >>> d = np.array([1.0, 1.0])
        >>> g = 0.1
        >>> K = covar_sep(col, X1, n1, X2, n2, d, g)
        >>> print(K)
        [[1.12535175e-07 3.72007598e-44]
         [3.72007598e-44 1.38389653e-87]
         [1.38389653e-87 5.14820022e-131]]
    """
    K = np.zeros((n1, n2))
    X1 = prepare_X(X1)
    X2 = prepare_X(X2)

    for i in range(n1):
        for j in range(n2):
            K[i, j] = 0.0
            for k in range(col):
                K[i, j] += (X1[i, k] - X2[j, k]) ** 2 / d[k]
            if i == j and K[i, j] == 0.0:
                K[i, j] = 1.0 + g
            else:
                K[i, j] = np.exp(0.0 - K[i, j])

    return K


def covar_sep_symm(col, X, n, d, g) -> np.ndarray:
    """
    Calculate the correlation (K) between X1 and X2 with a separable power exponential correlation function with range d and nugget g.

    Args:
        col (int): Number of columns in the input matrix X (features).
        X (ndarray): Input matrix of shape (n, col).
        n (int): Number of rows in the input matrix X.
        d (ndarray): Array of length col representing the range parameters, shape (col,).
        g (float): Nugget parameter.

    Returns:
        ndarray: The calculated covariance matrix K of shape (n, n).

    Examples:
        >>> from spotpython.gp.covar import covar_sep_symm
        >>> import numpy as np
        >>> col = 2
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> n = 3
        >>> d = np.array([1.0, 1.0])
        >>> g = 0.1
        >>> K = covar_sep_symm(col, X, n, d, g)
        >>> print(K)
        [[1.1        0.01831564 0.00012341]
         [0.01831564 1.1        0.01831564]
         [0.00012341 0.01831564 1.1       ]]
    """
    K = np.zeros((n, n))
    X = prepare_X(X)

    # calculate the covariance matrix K
    for i in range(n):
        K[i, i] = 1.0 + g
        for j in range(i + 1, n):
            K[i, j] = 0.0
            for k in range(col):
                K[i, j] += (X[i, k] - X[j, k]) ** 2 / d[k]
            K[i, j] = np.exp(-K[i, j])
            K[j, i] = K[i, j]

    return K


def diff_covar_sep(col, X1, n1, X2, n2, d, K) -> np.ndarray:
    """
    Calculate the first and second derivative (wrt d) of the correlation (K)
    between X1 and X2 with a separable power exponential correlation function
    with range d and nugget g (though g not needed).

    Args:
        col (int): Number of columns in the input matrices X1 and X2.
        X1 (ndarray): First input matrix of shape (n1, col).
        n1 (int): Number of rows in the first input matrix X1.
        X2 (ndarray): Second input matrix of shape (n2, col).
        n2 (int): Number of rows in the second input matrix X2.
        d (ndarray): Array of length col representing the range parameters.
        K (ndarray): Covariance matrix of shape (n1, n2).

    Returns:
        ndarray: The calculated derivative covariance matrix dK of shape (col, n1, n2).

    Examples:
        >>> col = 2
        >>> X1 = np.array([[1, 2], [3, 4], [5, 6]])
        >>> n1 = 3
        >>> X2 = np.array([[7, 8], [9, 10]])
        >>> n2 = 2
        >>> d = np.array([1.0, 1.0])
        >>> K = np.exp(-np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        >>> dK = diff_covar_sep(col, X1, n1, X2, n2, d, K)
        >>> print(dK)
        [[[1.12535175e-07 3.72007598e-44]
          [3.72007598e-44 1.38389653e-87]
          [1.38389653e-87 5.14820022e-131]]
         [[1.12535175e-07 3.72007598e-44]
          [3.72007598e-44 1.38389653e-87]
          [1.38389653e-87 5.14820022e-131]]]
    """
    X1 = prepare_X(X1)
    X2 = prepare_X(X2)
    dK = np.zeros((col, n1, n2))

    for k in range(col):
        d2k = d[k] ** 2
        for i in range(n1):
            for j in range(n2):
                dK[k, i, j] = K[i, j] * ((X1[i, k] - X2[j, k]) ** 2) / d2k

    return dK


def diff_covar_sep_symm(col, X, n, d, K) -> np.ndarray:
    """
    Calculate the first and second derivative (wrt d) of the correlation (K)
    between X1 and X2 with a separable power exponential correlation function
    with range d and nugget g (though g not needed) -- assumes symmetric matrix.

    Args:
        col (int): Number of columns in the input matrix X.
        X (ndarray): Input matrix of shape (n, col).
        n (int): Number of rows in the input matrix X.
        d (ndarray): Array of length col representing the range parameters.
        K (ndarray): Covariance matrix of shape (n, n).

    Returns:
        ndarray: The calculated derivative covariance matrix dK of shape (col, n, n).

    Examples:
        >>> col = 2
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> n = 3
        >>> d = np.array([1.0, 1.0])
        >>> K = np.exp(-np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]))
        >>> dK = diff_covar_sep_symm(col, X, n, d, K)
        >>> print(dK)
        [[[0.         0.36787944 0.01831564]
          [0.36787944 0.         0.36787944]
          [0.01831564 0.36787944 0.        ]]
         [[0.         0.36787944 0.01831564]
          [0.36787944 0.         0.36787944]
          [0.01831564 0.36787944 0.        ]]]
    """
    X = prepare_X(X)
    dK = np.zeros((col, n, n))

    for k in range(col):
        d2k = d[k] ** 2
        for i in range(n):
            for j in range(i + 1, n):
                dK[k, i, j] = dK[k, j, i] = K[i, j] * ((X[i, k] - X[j, k]) ** 2) / d2k
            dK[k, i, i] = 0.0

    return dK
