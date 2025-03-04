import numpy as np
from scipy.linalg import cho_factor, cho_solve


def linalg_dsymv(n, alpha, A, lda, x, incx, beta, y, incy) -> np.ndarray:
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


def linalg_ddot(n, x, incx, y, incy) -> float:
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


def linalg_dposv(n, Mutil, Mi) -> tuple:
    """
    Solve the linear equations A * x = B for x, where A is a symmetric positive definite matrix.

    Args:
        n (int): The order of the matrix Mutil.
        Mutil (ndarray): The matrix A.
        Mi (ndarray): The matrix B.

    Returns:
        tuple: The updated matrix Mi and an info flag.
            - Mi (ndarray): The solution matrix x.
            - Info flag (0 if successful, non-zero if an error occurred).

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

    return Mi, info
