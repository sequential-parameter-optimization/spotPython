import numpy as np
from numpy.linalg import det
from spotpython.gp.distances import covar_anisotropic, dist
from spotpython.utils.linalg import matrix_inversion_dispatcher


def nlsep(par, X, Y, nlsep_method="inv") -> float:
    """
    Calculate the negative log-likelihood for a separable power exponential correlation function.

    Args:
        par (np.ndarray):
            Array of parameters, where the first ncol(X) elements are the range parameters and the last element is the nugget parameter.
        X (np.ndarray):
            Input matrix of shape (n, col).
        Y (np.ndarray):
            Response vector of shape (n,).

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
    Ki = matrix_inversion_dispatcher(K, method=nlsep_method)
    detK = det(K)
    if detK <= 1e-14:
        detK = 1e-14  # TODO: Check if this can be improved
    ldetK = np.log(detK)
    ll = -(n / 2) * np.log(Y.T @ Ki @ Y) - (1 / 2) * ldetK
    return -ll


def nl(par, D, Y, nl_method="inv") -> float:
    """
    Calculate the negative log-likelihood for an exponential correlation function.

    Args:
        par (np.ndarray): Array of parameters, where the first element is the range parameter
                          and the second element is the nugget parameter.
        D (np.ndarray): Distance matrix of shape (n, n).
        Y (np.ndarray): Response vector of shape (n,).
        nl_method (str): The inversion method to use.

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
    Ki = matrix_inversion_dispatcher(K, method=nl_method)
    ldetK = np.log(det(K))
    ll = -(n / 2) * np.log(Y.T @ Ki @ Y) - (1 / 2) * ldetK
    return -ll


def gradnl(par, D, Y, gradnl_method="inv") -> np.ndarray:
    """
    Calculate the gradient of the negative log-likelihood for an exponential correlation function.

    Args:
        par (np.ndarray): Array of parameters, where the first element is the range parameter
                          and the second element is the nugget parameter.
        D (np.ndarray): Distance matrix of shape (n, n).
        Y (np.ndarray): Response vector of shape (n,).
        gradnl_method (str): The inversion method to use.

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
    Ki = matrix_inversion_dispatcher(K, method=gradnl_method)
    dotK = K * D / theta**2
    KiY = Ki @ Y

    # Theta component
    dlltheta = (n / 2) * (KiY.T @ dotK @ KiY) / (Y.T @ KiY) - (1 / 2) * np.sum(np.diag(Ki @ dotK))

    # G component
    dllg = (n / 2) * (KiY.T @ KiY) / (Y.T @ KiY) - (1 / 2) * np.sum(np.diag(Ki))

    # Combine the components into a gradient vector
    return -np.array([dlltheta, dllg])


def gradnlsep(par, X, Y, gradnlsep_method="inv") -> np.ndarray:
    """
    Compute gradient of the negative log-likelihood using full matrix inverse.

    Args:
        par (np.ndarray): Array of parameters, where the first ncol(X) elements are the range parameters
                          and the last element is the nugget parameter.
        X (np.ndarray): Input matrix of shape (n, col).
        Y (np.ndarray): Response vector of shape (n,).
        gradnlsep_method (str): The inversion method to use.

    """
    n_col = X.shape[1]
    if len(par) != n_col + 1:
        raise ValueError("Parameter size must be ncol(X) + 1.")
    theta = par[:n_col]
    g = par[n_col]
    n = len(Y)

    K = covar_anisotropic(X, d=theta, g=g)
    Ki = matrix_inversion_dispatcher(K, method=gradnlsep_method)
    KiY = Ki @ Y

    dlltheta = np.empty(len(theta))
    for k in range(len(dlltheta)):
        dotK = K * dist(X[:, [k]]) / (theta[k] ** 2)
        numerator = float(KiY.T @ dotK @ KiY)
        denominator = float(Y.T @ KiY)
        dlltheta[k] = (n / 2.0) * (numerator / denominator) - 0.5 * np.sum(np.diag(Ki @ dotK))

    numerator_g = float(KiY.T @ KiY)
    denominator_g = float(Y.T @ KiY)
    dllg = (n / 2.0) * (numerator_g / denominator_g) - 0.5 * np.sum(np.diag(Ki))
    return -np.concatenate([dlltheta, [dllg]])
