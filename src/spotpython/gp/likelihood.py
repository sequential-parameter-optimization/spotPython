import numpy as np
from numpy.linalg import inv, det
from spotpython.gp.distances import covar_anisotropic, dist
from scipy import linalg


def nlsep(par, X, Y, nlsep_method="inv") -> float:
    """
    Calculate the negative log-likelihood for a separable power exponential correlation function.

    Args:
        par (np.ndarray): Array of parameters, where the first ncol(X) elements are the range parameters
                          and the last element is the nugget parameter.
        X (np.ndarray): Input matrix of shape (n, col).
        Y (np.ndarray): Response vector of shape (n,).
        nlsep_method (str): Method name. Can be "inv" or "chol". Defaults to "inv".

    Returns:
        float: Negative log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.likelihood import nlsep
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> Y = np.array([1.0, 2.0, 3.0])
        >>> par = np.array([0.5, 0.5, 0.1])
        >>> result = nlsep(par, X, Y)
        >>> print(result)
        2.772588722239781

    """
    if nlsep_method == "chol":
        return nlsep_chol(par, X, Y)
    elif nlsep_method == "inv":
        return nlsep_inv(par, X, Y)
    else:
        raise ValueError("`nlsep_method` must be one of {'inv', 'chol'}.")


def nlsep_inv(par, X, Y) -> float:
    """
    Calculate the negative log-likelihood for a separable power exponential correlation function.

    Args:
        par (np.ndarray): Array of parameters, where the first ncol(X) elements are the range parameters
                          and the last element is the nugget parameter.
        X (np.ndarray): Input matrix of shape (n, col).
        Y (np.ndarray): Response vector of shape (n,).

    Returns:
        float: Negative log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.likelihood import nlsep
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> Y = np.array([1.0, 2.0, 3.0])
        >>> par = np.array([0.5, 0.5, 0.1])
        >>> result = nlsep(par, X, Y)
        >>> print(result)
        2.772588722239781
    """
    theta = par[: X.shape[1]]
    g = par[X.shape[1]]
    n = len(Y)
    K = covar_anisotropic(X, d=theta, g=g)
    Ki = inv(K)
    ldetK = np.log(det(K))
    ll = -(n / 2) * np.log(Y.T @ Ki @ Y) - (1 / 2) * ldetK
    return -ll


def nlsep_chol(par, X, Y) -> float:
    """
    Calculate the negative log-likelihood for a separable power exponential correlation function.

    Args:
        par (np.ndarray): Array of parameters, where the first ncol(X) elements are the range parameters
                          and the last element is the nugget parameter.
        X (np.ndarray): Input matrix of shape (n, col).
        Y (np.ndarray): Response vector of shape (n,).

    Returns:
        float: Negative log-likelihood.
    """
    theta = par[: X.shape[1]]
    g = par[X.shape[1]]

    print(f"theta: {theta}")
    print(f"g: {g}")

    n = len(Y)

    # Ensure g is at least a small positive value for numerical stability
    g = max(g, np.finfo(float).eps)

    # Calculate covariance matrix
    K = covar_anisotropic(X, d=theta, g=g)

    def try_cholesky(Kmat, max_attempts=3):
        """
        Attempt Cholesky on Kmat multiple times, increasing jitter at each step.
        Returns (ll, success) where ll is the log-likelihood contribution, and
        success indicates if Cholesky eventually succeeded.
        """
        jitter_scale = 1e-8
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    # Add jitter based on the trace
                    jitter = jitter_scale * np.trace(Kmat) / len(Kmat)
                    Kmat = Kmat + np.eye(len(Kmat)) * jitter
                L = linalg.cholesky(Kmat, lower=True)
                ldetK = 2.0 * np.sum(np.log(np.diag(L)))
                alpha = linalg.solve_triangular(L, Y, lower=True)
                quadform = np.sum(alpha**2)
                ll_val = -(n / 2.0) * np.log(quadform) - 0.5 * ldetK
                return (ll_val, True)
            except linalg.LinAlgError:
                jitter_scale *= 10.0  # Increase jitter by factor of 10
        return (None, False)

    # Try Cholesky decomposition with multi-jitter fallback
    ll, success = try_cholesky(K, max_attempts=3)
    if success:
        return -ll
    else:
        # Final fallback: direct approach with regularization
        # for a guaranteed SPD matrix, but less numerically stable
        jitter = 1e-6 * np.trace(K) / len(K)
        K_reg = K + np.eye(n) * jitter
        Ki = inv(K_reg)
        ldetK = np.log(max(det(K_reg), np.finfo(float).tiny))
        ll_final = -(n / 2.0) * np.log(Y.T @ Ki @ Y) - 0.5 * ldetK
        return -ll_final


def nl(par, D, Y) -> float:
    """
    Calculate the negative log-likelihood for an exponential correlation function.

    Args:
        par (np.ndarray): Array of parameters, where the first element is the range parameter
                          and the second element is the nugget parameter.
        D (np.ndarray): Distance matrix of shape (n, n).
        Y (np.ndarray): Response vector of shape (n,).

    Returns:
        float: Negative log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.likelihood import nl
        >>> D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        >>> Y = np.array([1.0, 2.0, 3.0])
        >>> par = np.array([0.5, 0.1])
        >>> result = nl(par, D, Y)
        >>> print(result)
        2.772
    """
    theta = par[0]  # change 1
    g = par[1]
    n = len(Y)
    K = np.exp(-D / theta) + np.diag([g] * n)  # change 2
    Ki = inv(K)
    ldetK = np.log(det(K))
    ll = -(n / 2) * np.log(Y.T @ Ki @ Y) - (1 / 2) * ldetK
    return -ll


def gradnl(par, D, Y) -> np.ndarray:
    """
    Calculate the gradient of the negative log-likelihood for an exponential correlation function.

    Args:
        par (np.ndarray): Array of parameters, where the first element is the range parameter
                          and the second element is the nugget parameter.
        D (np.ndarray): Distance matrix of shape (n, n).
        Y (np.ndarray): Response vector of shape (n,).

    Returns:
        np.ndarray: Gradient vector.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.likelihood import gradnl
        >>> D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        >>> Y = np.array([1.0, 2.0, 3.0])
        >>> par = np.array([0.5, 0.1])
        >>> grad = gradnl(par, D, Y)
        >>> print(grad)
        [-0.000 -0.000]
    """
    # Extract parameters
    theta = par[0]
    g = par[1]

    # Calculate covariance quantities from data and parameters
    n = len(Y)
    K = np.exp(-D / theta) + np.diag([g] * n)
    Ki = inv(K)
    dotK = K * D / theta**2
    KiY = Ki @ Y

    # Theta component
    dlltheta = (n / 2) * (KiY.T @ dotK @ KiY) / (Y.T @ KiY) - (1 / 2) * np.sum(np.diag(Ki @ dotK))

    # G component
    dllg = (n / 2) * (KiY.T @ KiY) / (Y.T @ KiY) - (1 / 2) * np.sum(np.diag(Ki))

    # Combine the components into a gradient vector
    return -np.array([dlltheta, dllg])


def gradnlsep_inv(par, X, Y) -> np.ndarray:
    """
    Inverse-based approach for computing the gradient of the negative
    log-likelihood for a separable power exponential correlation function.
    """
    n_col = X.shape[1]
    if len(par) != n_col + 1:
        raise ValueError("The number of elements in par should be equal to " "the number of columns in X + 1")
    theta = par[:n_col]
    g = par[n_col]
    n = len(Y)

    # Build covariance, then invert
    K = covar_anisotropic(X, d=theta, g=g)
    Ki = inv(K)
    KiY = Ki @ Y

    # Loop over each theta dimension
    dlltheta = np.empty(len(theta))
    for k in range(len(dlltheta)):
        # Distance matrix for just the k-th dimension
        dotK = K * dist(X[:, [k]]) / (theta[k] ** 2)
        numerator = (KiY.T @ dotK @ KiY).item()
        denominator = (Y.T @ KiY).item()
        dlltheta[k] = (n / 2.0) * (numerator / denominator) - 0.5 * np.sum(np.diag(Ki @ dotK))

    # Nugget
    numerator_g = (KiY.T @ KiY).item()
    denominator_g = (Y.T @ KiY).item()
    dllg = (n / 2.0) * (numerator_g / denominator_g) - 0.5 * np.sum(np.diag(Ki))

    return -np.concatenate([dlltheta, [dllg]])


def gradnlsep_chol(par, X, Y, direct: bool = False) -> np.ndarray:
    """
    Cholesky-based approach for computing the gradient of the negative
    log-likelihood for a separable power exponential correlation function,
    with jitter fallback if Cholesky fails.

    If direct=False (default), we explicitly form K^-1 from the Cholesky factor.
    If direct=True, we compute partial derivatives without explicitly forming K^-1.
    """
    n_col = X.shape[1]
    if len(par) != n_col + 1:
        raise ValueError("The number of elements in par must be ncol(X)+1")
    theta = par[:n_col]
    g = par[n_col]
    n = len(Y)

    # Ensure a minimal positive nugget
    g = max(g, np.finfo(float).eps)
    K = covar_anisotropic(X, d=theta, g=g)

    def try_cholesky(Kmat, max_attempts=3):
        jitter_scale = 1e-8
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    # Add jitter proportional to the trace
                    jitter = jitter_scale * np.trace(Kmat) / len(Kmat)
                    Kmat += np.eye(len(Kmat)) * jitter
                L_ = linalg.cholesky(Kmat, lower=True)
                return L_, True
            except linalg.LinAlgError:
                jitter_scale *= 10.0
        return None, False

    # Attempt Cholesky
    L, success = try_cholesky(K)
    if not success:
        # Final fallback: direct approach with some fixed jitter
        jitter = 1e-6 * np.trace(K) / len(K)
        K += np.eye(n) * jitter
        L = linalg.cholesky(K, lower=True)

    # If NOT using the direct partial-derivative approach, form Ki from L:
    if not direct:
        Id = np.eye(n)
        tmp = linalg.solve_triangular(L, Id, lower=True)
        Ki = linalg.solve_triangular(L.T, tmp, lower=False)
        KiY = Ki @ Y

        dlltheta = np.empty(len(theta))
        for k in range(len(dlltheta)):
            dotK = K * dist(X[:, [k]]) / (theta[k] ** 2)
            numerator = (KiY.T @ dotK @ KiY).item()
            denominator = (Y.T @ KiY).item()
            dlltheta[k] = (n / 2.0) * (numerator / denominator) - 0.5 * np.sum(np.diag(Ki @ dotK))

        # Nugget
        numerator_g = (KiY.T @ KiY).item()
        denominator_g = (Y.T @ KiY).item()
        dllg = (n / 2.0) * (numerator_g / denominator_g) - 0.5 * np.sum(np.diag(Ki))

        return -np.concatenate([dlltheta, [dllg]])
    else:
        # direct=True approach: compute without explicitly forming K^-1
        #
        # We compute K^-1 * Y efficiently via triangular solves
        alpha = linalg.solve_triangular(L, Y, lower=True)
        alpha2 = linalg.solve_triangular(L.T, alpha, lower=False)  # K^-1 * Y
        denom = float(Y.T @ alpha2)
        dlltheta = np.empty(len(theta))

        for k in range(len(dlltheta)):
            # Compute derivative of K with respect to theta[k]
            dotK = K * dist(X[:, [k]]) / (theta[k] ** 2)

            # Compute first term: (n/2) * (alpha2^T * dotK * alpha2) / (Y^T * alpha2)
            numerator = float(alpha2.T @ dotK @ alpha2)

            # Compute the trace term using a more efficient approach
            # trace(K^-1 * dotK) = sum of diagonal elements of K^-1 * dotK
            # We compute this more efficiently using matrix properties

            # Method: Use trace(AB) = sum_i sum_j A_ij * B_ji = sum_i (A @ B)_ii
            # We compute diagonal elements directly without forming full K^-1
            trace_val = 0.0

            # For large matrices, we could use a stochastic trace estimator
            # But for clarity and accuracy, we compute it directly
            for i in range(n):
                # For each column i of K^-1, solve via triangular solves
                e_i = np.zeros(n)
                e_i[i] = 1.0

                # K^-1 * e_i via two triangular solves
                temp = linalg.solve_triangular(L, e_i, lower=True)
                k_inv_col = linalg.solve_triangular(L.T, temp, lower=False)

                # Compute the i-th diagonal element of K^-1 * dotK
                # This is the dot product of the i-th row of K^-1 with the i-th column of dotK
                # Since K^-1 is symmetric, the i-th row = i-th column
                trace_val += np.dot(k_inv_col, dotK[:, i])

            dlltheta[k] = (n / 2.0) * (numerator / denom) - 0.5 * trace_val

        # For the nugget derivative, we need trace(K^-1)
        # This is the sum of diagonal elements of K^-1
        trace_k_inv = 0.0
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            temp = linalg.solve_triangular(L, e_i, lower=True)
            k_inv_col = linalg.solve_triangular(L.T, temp, lower=False)
            trace_k_inv += k_inv_col[i]  # Just need the diagonal element

        numerator_g = float(alpha2.T @ alpha2)
        dllg = (n / 2.0) * (numerator_g / denom) - 0.5 * trace_k_inv

        return -np.concatenate([dlltheta, [dllg]])


def gradnlsep(par, X, Y, gradnlsep_method="inv") -> np.ndarray:
    """
    A unified wrapper for gradient computation of the negative log-likelihood
    for a separable power exponential correlation function.

    The `method` argument chooses between:
      - "inv": standard inversion-based approach (gradnlsep_inv)
      - "chol": Cholesky-based approach with Ki = L^{-T} L^{-1}
      - "direct": Cholesky-based approach using repeated solves
                  without explicitly forming K^-1 (more complex, but memory-friendly)

    Args:
        par (np.ndarray): The model parameters, size ncol(X) + 1
        X (np.ndarray): (n, m) design matrix
        Y (np.ndarray): (n,) target values
        gradnlsep_method (str): Method name. Can be "inv", "chol" or "direct". Defaults to "inv".

    Returns:
        np.ndarray: Gradient of size (m + 1,), where m is number of columns in X.
    """
    if gradnlsep_method == "inv":
        return gradnlsep_inv(par, X, Y)
    elif gradnlsep_method == "chol":
        return gradnlsep_chol(par, X, Y, direct=False)
    elif gradnlsep_method == "direct":
        return gradnlsep_chol(par, X, Y, direct=True)
    else:
        raise ValueError("`method` must be one of {'inv', 'chol', 'direct'}.")
