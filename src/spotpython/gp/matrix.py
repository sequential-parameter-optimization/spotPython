import numpy as np


def new_dup_vector(vold, n) -> np.ndarray:
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
    v = dupv(v, vold, n)
    return v


def dupv(v, vold, n) -> np.ndarray:
    """
    Copies vold to v (assumes v has already been allocated).

    Args:
        v (ndarray): The array to copy to.
        vold (ndarray): The original array to copy from.
        n (int): The size of the arrays.

    Returns:
        ndarray: The updated array v.

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
    return v


def new_vector(n) -> np.ndarray:
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


def new_matrix_bones(v, n1, n2) -> np.ndarray:
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


def new_matrix(n1, n2) -> np.ndarray:
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


def sub_p_matrix(V, p, v, nrows, lenp, col_offset) -> np.ndarray:
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

    Returns:
        ndarray: The updated destination matrix V.

    Examples:
        >>> V = np.zeros((3, 5))
        >>> p = np.array([0, 2])
        >>> v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> nrows = 3
        >>> lenp = 2
        >>> col_offset = 1
        >>> V = sub_p_matrix(V, p, v, nrows, lenp, col_offset)
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
    return V


def sub_p_matrix_rows(V, p, v, ncols, lenp, row_offset) -> np.ndarray:
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

    Returns:
        ndarray: The updated destination matrix V.

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
    return V


def new_p_submatrix_rows(p, v, nrows, ncols, row_offset) -> np.ndarray:
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
        V = sub_p_matrix_rows(V, p, v, ncols, nrows, row_offset)
    return V


def dup_matrix(m, M, n1, n2) -> np.ndarray:
    """
    Copy the contents of matrix M to matrix m.

    Args:
        m (ndarray): The destination matrix.
        M (ndarray): The source matrix.
        n1 (int): The number of rows in the matrices.
        n2 (int): The number of columns in the matrices.

    Returns:
        ndarray: The updated destination matrix m.

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
    return m


def new_dup_matrix(M, n1, n2) -> np.ndarray:
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


def new_id_matrix(n) -> np.ndarray:
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
