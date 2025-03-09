import numpy as np
from spotpython.gp.covar import covar_sep, covar_sep_symm


def new_matrix_bones(data, rows, cols) -> np.ndarray:
    """
    Create a matrix view of the given data.

    Args:
        data (np.ndarray): Input data.
        rows (int): Number of rows.
        cols (int): Number of columns.

    Returns:
        np.ndarray: Matrix view of the input data.
    """
    return np.reshape(data, (rows, cols))


def covar_anisotropic(X1=None, X2=None, d=None, g=None) -> np.ndarray:
    """
    Calculate the separable covariance matrix between the rows of X1 and X2 with lengthscale d and nugget g.

    Args:
        X1 (np.ndarray): First input matrix.
        X2 (np.ndarray): Second input matrix (optional).
        d (np.ndarray): Array of lengthscale parameters.
        g (float): Nugget parameter.

    Returns:
        np.ndarray: Covariance matrix K.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.distances import covar_anisotropic
        >>> X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> d = np.array([1.0, 1.0])
        >>> g = 0.1
        >>> K_symm = covar_anisotropic(X1=X1, d=d, g=g)
        >>> print(K_symm)
        [[1.1        0.36787944]
         [0.36787944 1.1       ]]
        >>> K = covar_anisotropic(X1=X1, X2=X2, d=d, g=g)
        >>> print(K)
        [[0.00012341 0.00033546]
         [0.00033546 0.000911
    """
    # Convert pandas dataframes to numpy arrays
    X1 = np.asarray(X1)
    if X2 is not None:
        X2 = np.asarray(X2)
    if X1 is None:
        raise ValueError("X1 cannot be None")

    if not isinstance(X1, np.ndarray):
        if X2 is None:
            raise ValueError("X2 cannot be None in this context")
        m = X2.shape[1]
        X1 = np.reshape(X1, (-1, m))
    else:
        m = X1.shape[1]
    n1 = X1.shape[0]

    if len(d) != m:
        raise ValueError("bad d argument")
    if not isinstance(g, float):
        raise ValueError("bad g argument")

    if X2 is None:
        # Calculate K using covar_sep_symm_R
        # K = covar_sep_symm_R(m, X1.flatten(), n1, d, g)
        K = covar_sep_symm(m, X1, n1, d, g)
        return K
    else:
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("col dim mismatch for X1 & X2")

        X2 = np.asarray(X2)
        n2 = X2.shape[0]

        # Calculate K using covar_sep_R
        # K = covar_sep_R(m, X1.flatten(), n1, X2.flatten(), n2, d, g)
        K = covar_sep(m, X1, n1, X2, n2, d, g)
        return K


def dist(X1, X2=None) -> np.ndarray:
    """
    Calculate the distance matrix between the rows of X1 and X2, or between X1 and itself when X2 is None.

    Args:
        X1 (np.ndarray): First input matrix.
        X2 (np.ndarray, optional): Second input matrix.

    Returns:
        np.ndarray: Distance matrix D.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.distances import dist
        >>> X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> D_symm = dist(X1)
        >>> print(D_symm)
        [[ 0.  8.]
         [ 8.  0.]]
        >>> D = dist(X1, X2)
        >>> print(D)
            [[32.  8.]
            [18.  2.]]
    """
    # Coerce arguments and extract dimensions
    X1 = np.asarray(X1)
    n1, m = X1.shape

    if X2 is None:
        # Calculate D using distance_symm_R
        D = distance_symm_R(X1.flatten(), n1, m)
        return D
    else:
        # Coerce arguments and extract dimensions
        X2 = np.asarray(X2)
        n2 = X2.shape[0]

        # Check inputs
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("col dim mismatch for X1 & X2")

        # Calculate D using distance_R
        D = distance_R(X1.flatten(), n1, X2.flatten(), n2, m)
        return D


def distance_R(X1_in, n1_in, X2_in, n2_in, m_in) -> np.ndarray:
    """
    Calculate the distance matrix between the rows of X1 and X2.

    Args:
        X1_in (np.ndarray): First input matrix of shape (n1, m).
        n1_in (int): Number of rows in X1.
        X2_in (np.ndarray): Second input matrix of shape (n2, m).
        n2_in (int): Number of rows in X2.
        m_in (int): Number of columns (features).

    Returns:
        np.ndarray: Distance matrix D of shape (n1, n2).

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.distances import distance_R
        >>> X1_in = np.array([1.0, 2.0, 3.0, 4.0])
        >>> n1_in = 2
        >>> X2_in = np.array([5.0, 6.0, 7.0, 8.0])
        >>> n2_in = 2
        >>> m_in = 2
        >>> D_out = distance_R(X1_in, n1_in, X2_in, n2_in, m_in)
        >>> print(D_out)
        [[32.  8.]
         [18.  2.]]
    """
    # Make matrix bones
    X1 = np.reshape(X1_in, (n1_in, m_in))
    X2 = np.reshape(X2_in, (n2_in, m_in))
    D = distance(X1, n1_in, X2, n2_in, m_in)
    return D


def distance(X1, n1, X2, n2, m) -> np.ndarray:
    """
    Calculate the distance matrix (D) between X1 and X2.

    Args:
        X1 (np.ndarray): First input matrix of shape (n1, m).
        n1 (int): Number of rows in X1.
        X2 (np.ndarray): Second input matrix of shape (n2, m).
        n2 (int): Number of rows in X2.
        m (int): Number of columns (features).

    Returns:
        np.ndarray: Distance matrix D of shape (n1, n2).

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.distances import distance
        >>> X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> n1 = 2
        >>> X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        >>> n2 = 2
        >>> m = 2
        >>> D_out = distance(X1, n1, X2, n2, m)
        >>> print(D_out)
        [[32.  8.]
         [18.  2.]]
    """
    D = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            D[i, j] = 0.0
            for k in range(m):
                D[i, j] += (X1[i, k] - X2[j, k]) ** 2

    return D


def distance_symm_R(X_in, n_in, m_in) -> np.ndarray:
    """
    Calculate the distance matrix between the rows of X and itself, with output in the symmetric D_out matrix.

    Args:
        X_in (np.ndarray): Input matrix of shape (n, m).
        n_in (int): Number of rows in X.
        m_in (int): Number of columns (features).

    Returns:
        np.ndarray: Symmetric distance matrix D of shape (n, n).

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.distances import distance_symm_R
        >>> X_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> n_in = 3
        >>> m_in = 2
        >>> D_out = distance_symm_R(X_in, n_in, m_in)
        >>> print(D_out)
        [[ 0.  8. 32.]
         [ 8.  0.  8.]
         [32.  8.  0.]]
    """
    n = n_in
    m = m_in

    # Make matrix bones
    X = new_matrix_bones(X_in, n, m)
    D = np.zeros((n, n))

    # For each row of X and itself
    for i in range(n):
        D[i][i] = 0.0
        for j in range(i + 1, n):
            D[i][j] = 0.0
            for k in range(m):
                D[i][j] += (X[i][k] - X[j][k]) ** 2
            D[j][i] = D[i][j]

    return D
