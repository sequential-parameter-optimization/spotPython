import numpy as np


def log_determinant_chol(M) -> float:
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
