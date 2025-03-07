import numpy as np
from numpy.linalg import inv, det
from spotpython.gp.distances import covar_anisotropic, dist
from scipy import linalg


def nlsep(par, X, Y) -> float:
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


def nlsep_0(par, X, Y) -> float:
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

    try:
        # Use Cholesky decomposition for numerical stability when calculating log-determinant
        L = linalg.cholesky(K, lower=True)
        ldetK = 2.0 * np.sum(np.log(np.diag(L)))

        # Use the Cholesky factor for solving systems instead of explicit inverse
        # which is more stable and efficient
        alpha = linalg.solve_triangular(L, Y, lower=True)
        quadform = np.sum(alpha**2)

        ll = -(n / 2.0) * np.log(quadform) - 0.5 * ldetK
    except linalg.LinAlgError:
        # If Cholesky fails, fall back to a regularized approach
        # Add a small jitter to the diagonal to ensure positive definiteness
        jitter = 1e-8 * np.trace(K) / len(K)
        K_reg = K + np.eye(n) * jitter

        # Try again with the regularized matrix
        try:
            L = linalg.cholesky(K_reg, lower=True)
            ldetK = 2.0 * np.sum(np.log(np.diag(L)))
            alpha = linalg.solve_triangular(L, Y, lower=True)
            quadform = np.sum(alpha**2)
            ll = -(n / 2.0) * np.log(quadform) - 0.5 * ldetK
        except linalg.LinAlgError:
            # If that still fails, use a more direct but less numerically stable approach
            Ki = inv(K_reg)
            ldetK = np.log(max(det(K_reg), np.finfo(float).tiny))
            ll = -(n / 2.0) * np.log(Y.T @ Ki @ Y) - 0.5 * ldetK

    return -ll


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


def gradnlsep(par, X, Y) -> np.ndarray:
    """
    Calculate the gradient of the negative log-likelihood for a separable power exponential correlation function.

    Args:
        par (np.ndarray): Array of parameters, where the first ncol(X) elements are the range parameters
                          and the last element is the nugget parameter.
        X (np.ndarray): Input matrix of shape (n, col).
        Y (np.ndarray): Response vector of shape (n,).

    Returns:
        np.ndarray: Gradient vector.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.likelihood import gradnlsep
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> Y = np.array([1.0, 2.0, 3.0])
        >>> par = np.array([0.5, 0.5, 0.1])
        >>> grad = gradnlsep(par, X, Y)
        >>> print(grad)
        [-0.000 -0.000 -0.000]
    """
    n_col = X.shape[1]
    # par is an array of parameters, where the first ncol(X) elements are the range parameters theta
    # and the last element is the nugget parameter g. par has shape (ncol(X) + 1,)
    if len(par) != n_col + 1:
        raise ValueError("The number of elements in par should be equal to the number of columns in X + 1")
    # Extract the range parameters theta from par
    theta = par[:n_col]
    # Extract the nugget parameter g from par
    g = par[n_col]
    # Calculate the covariance matrix K using the anisotropic covariance function
    n = len(Y)
    K = covar_anisotropic(X, d=theta, g=g)
    Ki = inv(K)
    KiY = Ki @ Y

    # Loop over theta components
    dlltheta = np.empty(len(theta))
    for k in range(len(dlltheta)):
        dotK = K * dist(X[:, [k]]) / (theta[k] ** 2)
        # Use .item() to convert the (1,1) result into a scalar
        numerator = (KiY.T @ dotK @ KiY).item()
        denominator = (Y.T @ KiY).item()
        dlltheta[k] = (n / 2) * (numerator / denominator) - (1 / 2) * np.sum(np.diag(Ki @ dotK))

    # For g, also ensure the result is a scalar
    numerator_g = (KiY.T @ KiY).item()
    denominator_g = (Y.T @ KiY).item()
    dllg = (n / 2) * (numerator_g / denominator_g) - (1 / 2) * np.sum(np.diag(Ki))

    return -np.concatenate([dlltheta, [dllg]])
