import numpy as np
from numpy.linalg import inv
from scipy import linalg


def try_cholesky(Kmat, max_attempts=3):
    """Attempt Cholesky on Kmat multiple times, adding jitter at each step."""
    jitter_scale = 1e-8
    for _attempt in range(max_attempts):
        try:
            L_ = linalg.cholesky(Kmat, lower=True)
            return L_, True
        except linalg.LinAlgError:
            jitter = jitter_scale * np.trace(Kmat) / len(Kmat)
            Kmat += np.eye(len(Kmat)) * jitter
            jitter_scale *= 10.0
    return None, False


def matrix_inversion_dispatcher(K: np.ndarray, method: str = "inv") -> np.ndarray:
    """
    Returns the inverse of K using one of three methods:
    'inv' -> direct numpy.linalg.inv(K),
    'chol' -> Cholesky factorization (then forms K^-1),
    'direct' -> Cholesky factorization with repeated solves (still forms K^-1).

    Args:
        K (ndarray): The matrix to invert.
        method (str): The inversion method to use.

    Returns:
        ndarray: The inverse of K.

    Raises:
        ValueError: If method is not 'inv', 'chol', or 'direct'.

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.linalg import matrix_inversion_dispatcher
        >>> K = np.array([[1.0, 0.5], [0.5, 1.0]])
        >>> Ki = matrix_inversion_dispatcher(K, method="inv")
        >>> print(Ki)
        [[ 1.33333333 -0.66666667]
         [-0.66666667  1.33333333]]
    """
    n = K.shape[0]
    if method == "inv":
        return inv(K)

    L, success = try_cholesky(K.copy())
    if not success:
        # If Cholesky fails repeatedly, we can still do a naive inverse as fallback
        return inv(K)

    if method == "chol":
        # Build K^-1 from L
        Id = np.eye(n)
        tmp = linalg.solve_triangular(L, Id, lower=True)
        Ki = linalg.solve_triangular(L.T, tmp, lower=False)
        return Ki

    elif method == "direct":
        # Build K^-1 by repeated solves on identity columns
        Ki = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            # Solve K * x = e_i using the Cholesky factor
            tmp = linalg.solve_triangular(L, e_i, lower=True)
            x = linalg.solve_triangular(L.T, tmp, lower=False)
            Ki[:, i] = x
        return Ki

    else:
        raise ValueError("method must be 'inv', 'chol', or 'direct'.")
