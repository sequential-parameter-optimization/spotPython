import numpy as np

def random_orientation(k: int, p: int, xi: float) -> np.ndarray:
    """
    Generates a random orientation matrix for a screening design based on the Morris method.
    Translated from the original MATLAB function randorient.m.

    This function creates a (k+1) x k matrix that specifies a random path through the design space.
    Each element in the path is at a distance Delta = xi/(p-1) from one level to the next on a p-level grid.

    Args:
        k (int):
            Number of design variables.
        p (int):
            Number of discrete levels along each dimension.
        xi (float):
            Elementary effect step length factor.

    Returns:
        np.ndarray:
            A (k+1) x k matrix (Bstar) representing the randomized orientation.

    Examples:
        >>> import numpy as np
        >>> Bstar = random_orientation(k=3, p=4, xi=1.0)
        >>> print(Bstar)
    """
    # Step length
    Delta = xi / (p - 1)

    # Number of rows in the resulting orientation matrix
    m = k + 1

    # Create a truncated p-level grid in one dimension
    # up to (1 - Delta) in steps of 1/(p-1)
    step = 1.0 / (p - 1)
    xs = np.arange(0, 1.0 - Delta + step / 2, step)
    xsl = len(xs)

    # Build the basic sampling matrix B
    # Shape: (k+1, k)
    top_row = np.zeros((1, k))
    lower_tri = np.tril(np.ones((k, k)))
    B = np.vstack([top_row, lower_tri])

    # Matrix Dstar with +1 or -1 on the diagonal
    sign_vals = 2 * np.round(np.random.rand(k)) - 1
    Dstar = np.diag(sign_vals)

    # Random base value xstar for each column
    rand_indices = np.floor(np.random.rand(k) * xsl).astype(int)
    xstar = xs[rand_indices]

    # Permutation matrix Pstar of dimension k x k
    Pstar = np.zeros((k, k))
    perm = np.random.permutation(k)
    for i in range(k):
        Pstar[i, perm[i]] = 1

    # Construct the orientation matrix
    # (ones(m,1)*xstar) builds the base, then we add the randomized increments
    # and multiply with Pstar for column permutation
    ones_m_k = np.ones((m, k))
    partial_matrix = (2 * B - ones_m_k) @ Dstar + ones_m_k
    B_star = np.outer(np.ones(m), xstar) + (Delta / 2.0) * partial_matrix
    B_star = B_star @ Pstar

    return B_star