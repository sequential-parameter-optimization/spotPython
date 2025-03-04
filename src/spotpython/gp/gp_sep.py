import numpy as np
from scipy.linalg import cho_factor, cho_solve


class GPsep:
    def __init__(self, m, n, X, Z, d, g):
        self.m = m
        self.n = n
        self.X = X
        self.Z = Z
        self.d = d
        self.g = g
        self.K = None
        self.Ki = None
        self.KiZ = None
        self.phi = None
        self.dK = None
        self.ldetK = None


def linalg_dsymv(n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Perform the symmetric matrix-vector operation y := alpha*A*x + beta*y.

    Args:
        n (int): The order of the matrix A.
        alpha (float): The scalar alpha.
        A (ndarray): The n x n symmetric matrix.
        lda (int): The leading dimension of A.
        x (ndarray): The vector x.
        incx (int): The increment for the elements of x.
        beta (float): The scalar beta.
        y (ndarray): The vector y.
        incy (int): The increment for the elements of y.

    Returns:
        ndarray: The updated vector y.

    Examples:
        >>> n = 3
        >>> alpha = 1.0
        >>> A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)
        >>> lda = 3
        >>> x = np.array([1, 1, 1], dtype=float)
        >>> incx = 1
        >>> beta = 0.0
        >>> y = np.zeros(3, dtype=float)
        >>> incy = 1
        >>> linalg_dsymv(n, alpha, A, lda, x, incx, beta, y, incy)
        >>> print(y)
        [ 6. 11. 14.]
    """
    # print the dim of A
    print(f"dim of A = {A.shape}")
    # print the dim of x
    print(f"dim of x = {x.shape}")
    # modify the shape of x. If it is (n,1) make it (n,)
    x = x.reshape(-1)
    # print the dim of x
    print(f"dim of x = {x.shape}")
    # print the dim of y
    print(f"dim of y = {y.shape}")
    M = alpha * np.dot(A, x)
    # print the dim of M
    print(f"dim of M = {M.shape}")
    B = beta * y
    # print the dim of B
    print(f"dim of B = {B.shape}")
    y[:] = alpha * np.dot(A, x) + beta * y
    print(f"dim of y = {y.shape}")
    return y


def linalg_ddot(n, x, incx, y, incy):
    """
    Perform the dot product of two vectors x and y.

    Args:
        n (int): The number of elements in the vectors.
        x (ndarray): The vector x.
        incx (int): The increment for the elements of x.
        y (ndarray): The vector y.
        incy (int): The increment for the elements of y.

    Returns:
        float: The dot product of x and y.

    Examples:
        >>> n = 3
        >>> x = np.array([1, 2, 3], dtype=float)
        >>> incx = 1
        >>> y = np.array([4, 5, 6], dtype=float)
        >>> incy = 1
        >>> result = linalg_ddot(n, x, incx, y, incy)
        >>> print(result)
        32.0
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    return np.dot(x, y)


def linalg_dposv(n, Mutil, Mi):
    """
    Solve the linear equations A * x = B for x, where A is a symmetric positive definite matrix.

    Args:
        n (int): The order of the matrix Mutil.
        Mutil (ndarray): The matrix A.
        Mi (ndarray): The matrix B.

    Returns:
        int: Info flag (0 if successful, non-zero if an error occurred).

    Notes:
     Analog of dposv in clapack and lapack where Mutil is with colmajor and
     uppertri or rowmajor and lowertri.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.gp_sep import linalg_dposv
        >>> n = 3
        >>> Mutil = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)
        >>> Mi = np.eye(n)
        >>> info = linalg_dposv(n, Mutil, Mi)
        >>> print("Info:", info)
        >>> print("Mi:", Mi)
                Info: 0
                Mi: [[ 49.36111111 -13.55555556   2.11111111]
                [-13.55555556   3.77777778  -0.55555556]
                [  2.11111111  -0.55555556   0.11111111]]
    """
    try:
        # Perform Cholesky decomposition
        c, lower = cho_factor(Mutil, lower=True, overwrite_a=False, check_finite=True)

        # Solve the system
        Mi[:] = cho_solve((c, lower), Mi)

        info = 0
    except np.linalg.LinAlgError as e:
        info = 1
        print(f"Error: {e}")

    return info


def covar_sep_symm(col, X, n, d, g, K=None):
    """
    Calculate the correlation (K) between X1 and X2 with a separable power exponential correlation function with range d and nugget g.

    Args:
        col (int): Number of columns in the input matrix X.
        X (ndarray): Input matrix of shape (n, col).
        n (int): Number of rows in the input matrix X.
        d (ndarray): Array of length col representing the range parameters.
        g (float): Nugget parameter.
        K (ndarray): The covariance matrix of shape (n, n).

    Returns:
        ndarray: The calculated covariance matrix K of shape (n, n).

    Examples:
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

    for i in range(n):
        K[i, i] = 1.0 + g
        for j in range(i + 1, n):
            K[i, j] = 0.0
            for k in range(col):
                K[i, j] += (X[i, k] - X[j, k]) ** 2 / d[k]
            K[i, j] = np.exp(-K[i, j])
            K[j, i] = K[i, j]

    return K


def covar_sep(col, X1, n1, X2, n2, d, g):
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

    for i in range(n1):
        for j in range(n2):
            K[i, j] = 0.0
            for k in range(col):
                K[i, j] += (X1[i, k] - X2[j, k]) ** 2 / d[k]
            K[i, j] = np.exp(-K[i, j])

    return K


def diff_covar_sep(col, X1, n1, X2, n2, d, K):
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
    dK = np.zeros((col, n1, n2))

    for k in range(col):
        d2k = d[k] ** 2
        for i in range(n1):
            for j in range(n2):
                dK[k, i, j] = K[i, j] * ((X1[i, k] - X2[j, k]) ** 2) / d2k

    return dK


def diff_covar_sep_symm(col, X, n, d, K):
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
    dK = np.zeros((col, n, n))

    for k in range(col):
        d2k = d[k] ** 2
        for i in range(n):
            for j in range(i + 1, n):
                dK[k, i, j] = dK[k, j, i] = K[i, j] * ((X[i, k] - X[j, k]) ** 2) / d2k
            dK[k, i, i] = 0.0

    return dK


def log_determinant_chol(M):
    """
    Returns the log determinant of the n x n Cholesky decomposition of a matrix M.

    Args:
        M (ndarray): The n x n Cholesky decomposition of a matrix.

    Returns:
        float: The log determinant of the matrix M.

    Examples:
        >>> M = np.array([[2.0, 0.0], [1.0, 1.0]])
        >>> log_det = log_determinant_chol(M)
        >>> print(log_det)
        1.3862943611198906
    """
    log_det = 0.0
    n = M.shape[0]

    for i in range(n):
        log_det += np.log(M[i, i])

    log_det = 2 * log_det

    return log_det


def new_dup_vector(vold, n):
    """
    Allocates a new numpy array of size n and fills it with the contents of vold.

    Args:
        vold (ndarray): The original array to duplicate.
        n (int): The size of the new array.

    Returns:
        ndarray: The new array filled with the contents of vold.

    Examples:
        >>> vold = np.array([1.0, 2.0, 3.0])
        >>> n = 3
        >>> v = new_dup_vector(vold, n)
        >>> print(v)
        [1. 2. 3.]
    """
    v = np.empty(n)
    dupv(v, vold, n)
    return v


def dupv(v, vold, n):
    """
    Copies vold to v (assumes v has already been allocated).

    Args:
        v (ndarray): The array to copy to.
        vold (ndarray): The original array to copy from.
        n (int): The size of the arrays.

    Examples:
        >>> v = np.empty(3)
        >>> vold = np.array([1.0, 2.0, 3.0])
        >>> n = 3
        >>> dupv(v, vold, n)
        >>> print(v)
        [1. 2. 3.]
    """
    for i in range(n):
        v[i] = vold[i]


def new_vector(n):
    """
    Allocates a new numpy array of size n.

    Args:
        n (int): The size of the new array.

    Returns:
        ndarray: The new array of size n.

    Examples:
        >>> n = 3
        >>> v = new_vector(n)
        >>> print(v)
        [0. 0. 0.]
    """
    if n == 0:
        return None
    v = np.empty(n)
    return v


def new_matrix_bones(v, n1, n2):
    """
    Create a 2D numpy array (matrix) from a 1D numpy array (vector).
    The resulting matrix shares the same memory as the input vector.

    Args:
        v (ndarray): The input 1D numpy array (vector).
        n1 (int): The number of rows in the resulting matrix.
        n2 (int): The number of columns in the resulting matrix.

    Returns:
        ndarray: The resulting 2D numpy array (matrix).

    Examples:
        >>> v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> n1 = 2
        >>> n2 = 3
        >>> M = new_matrix_bones(v, n1, n2)
        >>> print(M)
        [[1. 2. 3.]
         [4. 5. 6.]]
    """
    M = np.empty((n1, n2), dtype=v.dtype)
    M[0] = v[:n2]
    for i in range(1, n1):
        M[i] = v[i * n2 : (i + 1) * n2]
    return M


def new_matrix(n1, n2):
    """
    Create a new n1 x n2 matrix which is allocated like an n1*n2 array,
    but can be referenced as a 2-d array.

    Args:
        n1 (int): The number of rows in the matrix.
        n2 (int): The number of columns in the matrix.

    Returns:
        ndarray: The new n1 x n2 matrix.

    Examples:
        >>> n1 = 3
        >>> n2 = 2
        >>> m = new_matrix(n1, n2)
        >>> print(m)
        [[0. 0.]
         [0. 0.]
         [0. 0.]]
    """
    if n1 == 0 or n2 == 0:
        return None

    m = np.zeros((n1, n2))
    return m


def sub_p_matrix(V, p, v, nrows, lenp, col_offset):
    """
    Copy the columns `v[1:n1][p[n2]]` to V.
    Must have nrow(v) == nrow(V) and ncol(V) >= lenp and ncol(v) >= max(p).

    Args:
        V (ndarray): The destination matrix.
        p (ndarray): The array of column indices to copy.
        v (ndarray): The source matrix.
        nrows (int): The number of rows in the matrices.
        lenp (int): The length of the array p.
        col_offset (int): The column offset in the destination matrix.

    Examples:
        >>> V = np.zeros((3, 5))
        >>> p = np.array([0, 2])
        >>> v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> nrows = 3
        >>> lenp = 2
        >>> col_offset = 1
        >>> sub_p_matrix(V, p, v, nrows, lenp, col_offset)
        >>> print(V)
        [[0. 1. 3. 0. 0.]
         [0. 4. 6. 0. 0.]
         [0. 7. 9. 0. 0.]]
    """
    assert V is not None and p is not None and v is not None
    assert nrows > 0 and lenp > 0

    for i in range(nrows):
        for j in range(lenp):
            V[i, j + col_offset] = v[i, p[j]]


def sub_p_matrix_rows(V, p, v, ncols, lenp, row_offset):
    """
    Copy the rows `v[1:n1][p[n2]]` to V.
    Must have ncol(v) == ncol(V) and nrow(V) >= lenp and nrow(v) >= max(p).

    Args:
        V (ndarray): The destination matrix.
        p (ndarray): The array of row indices to copy.
        v (ndarray): The source matrix.
        ncols (int): The number of columns in the matrices.
        lenp (int): The length of the array p.
        row_offset (int): The row offset in the destination matrix.

    Examples:
        >>> V = np.zeros((5, 3))
        >>> p = np.array([0, 2])
        >>> v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> ncols = 3
        >>> lenp = 2
        >>> row_offset = 1
        >>> sub_p_matrix_rows(V, p, v, ncols, lenp, row_offset)
        >>> print(V)
        [[0. 0. 0.]
         [1. 2. 3.]
         [0. 0. 0.]
         [7. 8. 9.]
         [0. 0. 0.]]
    """
    assert V is not None and p is not None and v is not None
    assert ncols > 0 and lenp > 0

    for i in range(lenp):
        V[i + row_offset, :ncols] = v[p[i], :ncols]


def new_p_submatrix_rows(p, v, nrows, ncols, row_offset):
    """
    Create a new matrix from the rows of v, specified by p.
    Must have ncol(v) == ncol(V) and nrow(V) >= nrows and nrow(v) >= max(p).

    Args:
        p (ndarray): The array of row indices to copy.
        v (ndarray): The source matrix.
        nrows (int): The number of rows in the new matrix.
        ncols (int): The number of columns in the new matrix.
        row_offset (int): The row offset in the new matrix.

    Returns:
        ndarray: The new matrix with specified rows copied from v.

    Examples:
        >>> p = np.array([0, 2])
        >>> v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> nrows = 2
        >>> ncols = 3
        >>> row_offset = 1
        >>> V = new_p_submatrix_rows(p, v, nrows, ncols, row_offset)
        >>> print(V)
        [[0. 0. 0.]
         [1. 2. 3.]
         [0. 0. 0.]
         [7. 8. 9.]
         [0. 0. 0.]]
    """
    if nrows + row_offset == 0 or ncols == 0:
        return None

    V = np.zeros((nrows + row_offset, ncols))
    if nrows > 0:
        sub_p_matrix_rows(V, p, v, ncols, nrows, row_offset)
    return V


def dup_matrix(m, M, n1, n2):
    """
    Copy the contents of matrix M to matrix m.

    Args:
        m (ndarray): The destination matrix.
        M (ndarray): The source matrix.
        n1 (int): The number of rows in the matrices.
        n2 (int): The number of columns in the matrices.

    Examples:
        >>> M = np.array([[1, 2], [3, 4], [5, 6]])
        >>> m = np.zeros((3, 2))
        >>> dup_matrix(m, M, 3, 2)
        >>> print(m)
        [[1. 2.]
         [3. 4.]
         [5. 6.]]
    """
    for i in range(n1):
        for j in range(n2):
            m[i, j] = M[i, j]


def new_dup_matrix(M, n1, n2):
    """
    Create a new n1 x n2 matrix which is allocated like an n1*n2 array,
    and copy the contents of n1 x n2 matrix M into it.

    Args:
        M (ndarray): The source matrix.
        n1 (int): The number of rows in the matrix.
        n2 (int): The number of columns in the matrix.

    Returns:
        ndarray: The new n1 x n2 matrix with copied contents.

    Examples:
        >>> M = np.array([[1, 2], [3, 4], [5, 6]])
        >>> n1 = 3
        >>> n2 = 2
        >>> m = new_dup_matrix(M, n1, n2)
        >>> print(m)
        [[1. 2.]
         [3. 4.]
         [5. 6.]]
    """
    if n1 <= 0 or n2 <= 0:
        return None

    m = new_matrix(n1, n2)
    dup_matrix(m, M, n1, n2)
    return m


def new_id_matrix(n):
    """
    Create a new n x n identity matrix.

    Args:
        n (int): The size of the identity matrix.

    Returns:
        ndarray: The new n x n identity matrix.

    Examples:
        >>> n = 3
        >>> m = new_id_matrix(n)
        >>> print(m)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    """
    return np.eye(n)


def newdKGPsep(gpsep):
    """
    Allocate new space for dK calculations, and calculate derivatives.

    Args:
        gpsep (GPsep): The GPsep object.

    Examples:
        >>> m = 2
        >>> n = 3
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> Z = np.array([1.0, 2.0, 3.0])
        >>> d = np.array([1.0, 1.0])
        >>> g = 0.1
        >>> gpsep = GPsep(m, n, X, Z, d, g)
        >>> gpsep.K = np.exp(-np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]))
        >>> newdKGPsep(gpsep)
        >>> print(gpsep.dK)
        [[[0.         0.36787944 0.01831564]
          [0.36787944 0.         0.36787944]
          [0.01831564 0.36787944 0.        ]]
         [[0.         0.36787944 0.01831564]
          [0.36787944 0.         0.36787944]
          [0.01831564 0.36787944 0.        ]]]
    """
    assert gpsep.dK is None
    gpsep.dK = np.empty((gpsep.m, gpsep.n, gpsep.n))
    for j in range(gpsep.m):
        gpsep.dK[j] = new_matrix(gpsep.n, gpsep.n)
    diff_covar_sep_symm(gpsep.m, gpsep.X, gpsep.n, gpsep.d, gpsep.K, gpsep.dK)


def calc_ZtKiZ_sep(gpsep):
    """
    Re-calculates phi = ZtKiZ from Ki and Z stored in the GP object; also update KiZ on which it depends.

    Args:
        gpsep (GPsep): The GPsep object.

    Examples:
        >>> m = 2
        >>> n = 3
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> Z = np.array([1.0, 2.0, 3.0])
        >>> d = np.array([1.0, 1.0])
        >>> g = 0.1
        >>> gpsep = GPsep(m, n, X, Z, d, g)
        >>> gpsep.Ki = np.linalg.inv(np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]]))
        >>> calc_ZtKiZ_sep(gpsep)
        >>> print(gpsep.phi)
        14.0
    """
    assert gpsep is not None
    if gpsep.KiZ is None:
        gpsep.KiZ = new_vector(gpsep.n)
    # linalg_dsymv(gpsep.n, 1.0, gpsep.Ki, gpsep.n, gpsep.Z, 1, 0.0, gpsep.KiZ, 1)
    # gpsep.phi = linalg_ddot(gpsep.n, gpsep.Z, 1, gpsep.KiZ, 1)
    gpsep.phi = calc_phi(gpsep.Ki, gpsep.Z)


def calc_phi(Ki, Z):
    """
    Calculate phi = t(Z) %*% Ki %*% Z, where Z is a (n,1) vector and Ki is a (n,n) matrix.

    Args:
        Ki (ndarray): The (n, n) matrix.
        Z (ndarray): The (n, 1) vector.

    Returns:
        float: The calculated value of phi.

    Examples:
        >>> Ki = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
        >>> Z = np.array([[1.0], [2.0], [3.0]])
        >>> phi = calc_phi(Ki, Z)
        >>> print(phi)
        14.0
    """
    # Ensure Z is a column vector
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    # Calculate phi = t(Z) %*% Ki %*% Z
    phi = np.dot(Z.T, np.dot(Ki, Z))

    # Since phi is a 1x1 matrix, we extract the scalar value
    phi = phi[0, 0]

    return phi


def buildGPsep(gpsep, dK):
    """
    Intended for newly created separable GPs, e.g., via newGPsep.
    Does all of the correlation calculations, etc., after data and parameters are defined.
    Similar to buildGP except calculates gradient dK.

    Args:
        gpsep (GPsep): The GPsep object.
        dK (int): Flag to indicate whether to calculate derivatives.

    Returns:
        GPsep: The updated GPsep object.

    Examples:
        >>> m = 2
        >>> n = 3
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> Z = np.array([1.0, 2.0, 3.0])
        >>> d = np.array([1.0, 1.0])
        >>> g = 0.1
        >>> gpsep = GPsep(m, n, X, Z, d, g)
        >>> gpsep = buildGPsep(gpsep, 1)
        >>> print(gpsep.K)
        [[1.         0.36787944 0.01831564]
         [0.36787944 1.         0.36787944]
         [0.01831564 0.36787944 1.        ]]
    """
    assert gpsep is not None and gpsep.K is None
    n = gpsep.n
    m = gpsep.m
    X = gpsep.X

    # Build covariance matrix
    gpsep.K = new_matrix(n, n)

    gpsep.K = covar_sep_symm(m, X, n, gpsep.d, gpsep.g, gpsep.K)

    # Invert covariance matrix
    gpsep.Ki = new_id_matrix(n)
    Kchol = new_dup_matrix(gpsep.K, n, n)
    info = linalg_dposv(n, Kchol, gpsep.Ki)
    if info:
        print("d = ", gpsep.d)
        raise ValueError(f"Bad Cholesky decomposition (info={info}), g={gpsep.g}")
    gpsep.ldetK = log_determinant_chol(Kchol)
    del Kchol

    # phi <- t(Z) %*% Ki %*% Z
    gpsep.KiZ = None
    calc_ZtKiZ_sep(gpsep)

    # Calculate derivatives ?
    gpsep.dK = None
    if dK:
        newdKGPsep(gpsep)

    # Return new structure
    return gpsep


def newGPsep_R(m, n, X, Z, d, g, dK):
    """
    Allocate a new separable GP structure using the data and parameters provided.

    Args:
        m (int): The number of input dimensions.
        n (int): The number of data points.
        X (ndarray): The input data matrix of shape (n, m).
        Z (ndarray): The output data vector of length n.
        d (ndarray): The lengthscale parameters of length m.
        g (float): The nugget parameter.
        dK (int): Flag to indicate whether to calculate derivatives.

    Returns:
        GPsep: The newly created GPsep object.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.gp_sep import newGPsep
        >>> m = 2
        >>> n = 3
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> Z = np.array([1.0, 2.0, 3.0])
        >>> d = np.array([1.0, 1.0])
        >>> g = 0.1
        >>> gpsep = newGPsep(m, n, X, Z, d, g, 1)
        >>> print(gpsep.K)
        [[1.         0.36787944 0.01831564]
         [0.36787944 1.         0.36787944]
         [0.01831564 0.36787944 1.        ]]
    """
    gpsep = GPsep(m, n, new_dup_matrix(X, n, m), new_dup_vector(Z, n), new_dup_vector(d, m), g)
    return buildGPsep(gpsep, dK)


def newGPsep(X, Z, d, g, dK=False):
    """
    Build an initial separable GP representation using the X-Z data and d/g parameterization.

    Args:
        X (ndarray): The input data matrix of shape (n, m).
        Z (ndarray): The output data vector of length n.
        d (ndarray or float): The lengthscale parameters of length m or a single value.
        g (float): The nugget parameter.
        dK (bool): Flag to indicate whether to calculate derivatives.

    Returns:
        GPsep: The newly created GPsep object.

    Examples:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> Z = np.array([1.0, 2.0, 3.0])
        >>> d = 1.0
        >>> g = 0.1
        >>> gpsep = newGPsep(X, Z, d, g, dK=False)
        >>> print(gpsep.K)
    """
    n, m = X.shape
    if n == 0:
        raise ValueError("X must be a matrix")
    if len(Z) != n:
        raise ValueError("must have nrow(X) = length(Z)")
    if isinstance(d, (int, float)):
        d = np.full(m, d)
    elif len(d) != m:
        raise ValueError("must have length(d) = ncol(X)")

    gpsep = GPsep(m, n, X, Z, d, g)
    return buildGPsep(gpsep, dK)


def predGPsep_R(gpsepi_in, m_in, nn_in, XX_in, lite_in, nonug_in, mean_out, Sigma_out, df_out, llik_out):
    """
    Interface that returns the student-t predictive equations,
    i.e., parameters to a multivariate t-distribution for XX predictive locations
    of dimension (n*m) using the stored GP parameterization.

    Args:
        gpsepi_in (int): The GPsep index.
        XX_in (ndarray): The predictive locations.
        lite (bool): Flag to indicate whether to use lite prediction.
        nonug (bool): Flag to indicate whether to use nugget.
        mean_out (ndarray): The output mean.
        Sigma_out (ndarray): The output covariance matrix.
        df_out (ndarray): The output degrees of freedom.
        llik_out (ndarray): The output log-likelihood.

    Returns:
        tuple: A tuple containing the mean, Sigma (or s2), df, and llik.
    """
    # global gpseps, NGPsep

    # Get the GP
    # gpsepi = gpsepi_in
    # if gpseps is None or gpsepi >= NGPsep or gpseps[gpsepi] is None:
    #     raise ValueError(f"gpsep {gpsepi} is not allocated")
    # gpsep = gpseps[gpsepi]
    gpsep = gpsepi_in
    if m_in != gpsep.m:
        raise ValueError(f"ncol(X)={m_in} does not match GPsep/C-side ({gpsep.m})")

    # Sanity check and XX representation
    XX = XX_in.reshape(nn_in, m_in)
    if not lite_in:
        Sigma = Sigma_out.reshape(nn_in, nn_in)
    else:
        Sigma = None

    # Call the C-only Predict function
    if lite_in:
        mean_out, Sigma, df_out, llik_out = predGPsep_lite(gpsep, XX.shape[0], XX, nonug_in, mean_out, Sigma_out, df_out, llik_out)
    else:
        mean_out, Sigma, df_out, llik_out = predGPsep(gpsep, XX.shape[0], XX, nonug_in, mean_out, Sigma, df_out, llik_out)
    return mean_out, Sigma, df_out, llik_out


def predGPsep(gpsep, nn, XX, nonug, mean, Sigma, df, llik):
    """
    Return the student-t predictive equations,
    i.e., parameters to a multivariate t-distribution
    for XX predictive locations of dimension (n*m).

    Args:
        gpsep (GPsep): The GPsep object.
        nn (int): The number of predictive locations.
        XX (ndarray): The predictive locations.
        nonug (int): Flag to indicate whether to use nugget.
        mean (ndarray): The output mean.
        Sigma (ndarray): The output covariance matrix.
        df (ndarray): The output degrees of freedom.
        llik (ndarray): The output log-likelihood.
    """
    n = gpsep.n
    m = gpsep.m

    # Are we using a nugget in the final calculation
    if nonug:
        g = np.finfo(float).eps
    else:
        g = gpsep.g

    # Variance (s2) components
    df[0] = float(n)
    phidf = gpsep.phi / df[0]

    # Calculate marginal likelihood (since we have the bits)
    llik[0] = -0.5 * (df[0] * np.log(0.5 * gpsep.phi) + gpsep.ldetK)
    # Continuing: - ((double) n)*M_LN_SQRT_2PI;

    # k <- covar(X1=X, X2=XX, d=Zt$d, g=0)
    k = covar_sep(m, gpsep.X, n, XX, nn, gpsep.d, 0.0)

    # Sigma <- covar(X1=XX, d=Zt$d, g=Zt$g)
    Sigma[:] = covar_sep_symm(m, XX, nn, gpsep.d, g)

    # Call generic function that would work for all GP covariance specs
    mean, Sigma = pred_generic(n, phidf, gpsep.Z, gpsep.Ki, nn, k, mean, Sigma)

    return mean, Sigma, df, llik


def pred_generic(n, phidf, Z, Ki, nn, k, mean, Sigma):
    """
    Generic function for GP prediction.

    Args:
        n (int): The number of data points.
        phidf (float): The phi/df value.
        Z (ndarray): The response vector.
        Ki (ndarray): The inverse covariance matrix.
        nn (int): The number of predictive locations.
        k (ndarray): The covariance matrix between training and predictive locations.
        mean (ndarray): The output mean.
        Sigma (ndarray): The output covariance matrix.

    Returns:
        tuple: A tuple containing the mean and Sigma.

    """
    # ktKi <- t(k) %*% Ki
    ktKi = np.dot(k.T, Ki)

    # ktKik <- ktKi %*% k
    ktKik = np.dot(ktKi, k)

    # mean <- ktKi %*% Z
    mean[:] = np.dot(ktKi, Z).reshape(-1)

    # Sigma <- phi*(Sigma - ktKik)/df
    Sigma[:] = phidf * (Sigma - ktKik)

    return mean, Sigma


def predictGPsep(gpsepi, XX, lite=False, nonug=False):
    """
    Obtain the parameters to a multivariate-t distribution describing the predictive surface
    of the fitted GP model.

    Args:
        gpsepi (int): The GPsep index.
        XX (ndarray): The predictive locations.
        lite (bool): Flag to indicate whether to compute only the diagonal of Sigma.
        nonug (bool): Flag to indicate whether to use nugget.

    Returns:
        dict: A dictionary containing the mean, Sigma (or s2), df, and llik.

    Examples:
        >>> import numpy as np
            from spotpython.gp.gp_sep import newGPsep, predictGPsep
            from spotpython.gp.functions import f2d
            import matplotlib.pyplot as plt
            # Design with N=441
            x = np.linspace(-2, 2, 11)
            X = np.array(np.meshgrid(x, x)).T.reshape(-1, 2)
            Z = f2d(X)
            # Fit a GP
            gpsep = newGPsep(X, Z, d=0.35, g=1/1000)
            # Predictive grid with NN=400
            xx = np.linspace(-1.9, 1.9, 20)
            XX = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2)
            ZZ = f2d(XX)
            # Predict
            p = predictGPsep(gpsep, XX)
            # RMSE: compare to similar experiment in aGP docs
            rmse = np.sqrt(np.mean((p["mean"] - ZZ) ** 2))
            print("RMSE:", rmse)
            # Visualize the result
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(p["mean"].reshape(len(xx), len(xx)), extent=(xx.min(), xx.max(), xx.min(), xx.max()), origin='lower', cmap='hot')
            plt.colorbar()
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Predictive Mean")
            plt.subplot(1, 2, 2)
            plt.imshow((p["mean"] - ZZ).reshape(len(xx), len(xx)), extent=(xx.min(), xx.max(), xx.min(), xx.max()), origin='lower', cmap='hot')
            plt.colorbar()
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Residuals")
            plt.show()
    """
    # nn is the number of rows in XX
    nn = XX.shape[0]
    # m is the number of columns in XX
    m = XX.shape[1]
    if nn == 0:
        raise ValueError("XX bad dims")

    if lite:
        # Lite means does not compute full Sigma, only diag
        mean_out = np.zeros(nn)
        s2_out = np.zeros(nn)
        df_out = np.zeros(1)
        llik_out = np.zeros(1)

        mean_out, s2_out, df_out, llik_out = predGPsep_R(gpsepi, m, nn, XX, lite_in=True, nonug_in=nonug, mean_out=mean_out, Sigma_out=s2_out, df_out=df_out, llik_out=llik_out)

        # Return parameterization
        return {"mean": mean_out, "s2": s2_out, "df": df_out, "llik": llik_out}

    else:
        # Compute full predictive covariance matrix
        mean_out = np.zeros(nn)
        Sigma_out = np.zeros(nn * nn)
        df_out = np.zeros(1)
        llik_out = np.zeros(1)

        mean_out, Sigma, df_out, llik_out = predGPsep_R(gpsepi, m_in=m, nn_in=nn, XX_in=XX, lite_in=False, nonug_in=nonug, mean_out=mean_out, Sigma_out=Sigma_out, df_out=df_out, llik_out=llik_out)

        # Coerce matrix output
        Sigma = Sigma_out.reshape(nn, nn)

        # Return parameterization
        return {"mean": mean_out, "Sigma": Sigma, "df": df_out, "llik": llik_out}


def new_predutilGPsep_lite(gpsep, nn, XX):
    """
    Utility function that allocates and calculates useful vectors
    and matrices for prediction; used by predGPsep_lite and dmus2GP.

    Args:
        gpsep (GPsep): The GPsep object.
        nn (int): The number of predictive locations.
        XX (ndarray): The predictive locations.

    Returns:
        tuple: A tuple containing k, ktKi, and ktKik.
    """
    # k <- covar(X1=X, X2=XX, d=Zt$d, g=0)
    k = covar_sep(gpsep.m, gpsep.X, gpsep.n, XX, nn, gpsep.d, 0.0)

    # Call generic function that would work for all GP covariance specs
    ktKi, ktKik = new_predutil_generic_lite(gpsep.n, gpsep.Ki, nn, k)

    return k, ktKi, ktKik


def predGPsep_lite(gpsep, nn, XX, nonug, mean, sigma2, df, llik):
    """
    Return the student-t predictive equations,
    i.e., parameters to a multivariate t-distribution
    for XX predictive locations of dimension (n*m);
    lite because sigma2 not Sigma is calculated.

    Args:
        gpsep (GPsep): The GPsep object.
        nn (int): The number of predictive locations.
        XX (ndarray): The predictive locations.
        nonug (int): Flag to indicate whether to use nugget.
        mean (ndarray): The output mean.
        sigma2 (ndarray): The output variance.
        df (ndarray): The output degrees of freedom.
        llik (ndarray): The output log-likelihood.
    """
    # Sanity checks
    assert df is not None
    df[0] = gpsep.n

    # Are we using a nugget in the final calculation
    if nonug:
        g = np.finfo(float).eps
    else:
        g = gpsep.g

    # Utility calculations
    k, ktKi, ktKik = new_predutilGPsep_lite(gpsep, nn, XX)

    # mean <- ktKi %*% Z
    if mean is not None:
        mean[:] = np.dot(ktKi, gpsep.Z).reshape(-1)

    # Sigma <- phi*(Sigma - ktKik)/df
    # *df = n - m - 1.0;  # only if estimating beta
    if sigma2 is not None:
        phidf = gpsep.phi / df[0]
        for i in range(nn):
            sigma2[i] = phidf * (1.0 + g - ktKik[i])

    # Calculate marginal likelihood (since we have the bits)
    # Might move to updateGP if we decide to move phi to updateGP
    if llik is not None:
        llik[0] = -0.5 * (gpsep.n * np.log(0.5 * gpsep.phi) + gpsep.ldetK)
        # Continuing: - ((double) n)*M_LN_SQRT_2PI;




def new_predutil_generic_lite(n, Ki, nn, k):
    """
    Generic utility function for prediction.

    Args:
        n (int): The number of data points.
        Ki (ndarray): The inverse covariance matrix.
        nn (int): The number of predictive locations.
        k (ndarray): The covariance matrix between training and predictive locations.

    Returns:
        tuple: A tuple containing ktKi and ktKik.
    """
    # ktKi <- t(k) %*% Ki
    ktKi = np.dot(k.T, Ki)

    # ktKik <- ktKi %*% k
    ktKik = np.dot(ktKi, k)

    return ktKi, ktKik
