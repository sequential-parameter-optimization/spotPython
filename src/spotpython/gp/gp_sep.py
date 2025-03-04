import numpy as np
from spotpython.gp.linalg import linalg_dposv
from spotpython.gp.covar import covar_sep_symm, covar_sep, diff_covar_sep_symm
from spotpython.gp.util import log_determinant_chol
from spotpython.gp.matrix import new_matrix, new_vector, new_id_matrix, new_dup_matrix
from spotpython.gp.lite import predGPsep_lite


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


def newdKGPsep(gpsep) -> GPsep:
    """
    Allocate new space for dK calculations, and calculate derivatives.
    Updates the dK and dk fields of the GPsep object.

    Args:
        gpsep (GPsep): The GPsep object.

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
    gpsep.dk = diff_covar_sep_symm(gpsep.m, gpsep.X, gpsep.n, gpsep.d, gpsep.K, gpsep.dK)
    return gpsep


def calc_ZtKiZ_sep(gpsep) -> GPsep:
    """
    Re-calculates phi = ZtKiZ from Ki and Z stored in the GP object; also update KiZ on which it depends.

    Args:
        gpsep (GPsep): The GPsep object.

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
        >>> gpsep.Ki = np.linalg.inv(np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]]))
        >>> calc_ZtKiZ_sep(gpsep)
        >>> print(gpsep.phi)
        14.0
    """
    assert gpsep is not None
    if gpsep.KiZ is None:
        gpsep.KiZ = new_vector(gpsep.n)

    # Ensure Z is a column vector
    Z = gpsep.Z
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    # Calculate phi = t(Z) %*% Ki %*% Z
    KiZ = np.dot(gpsep.Ki, Z)
    phi = np.dot(Z.T, KiZ)

    # Since phi is a 1x1 matrix, we extract the scalar value
    phi = phi[0, 0]

    gpsep.phi = phi
    gpsep.KiZ = KiZ
    return gpsep


def buildGPsep(gpsep, dK) -> GPsep:
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
    gpsep.Ki, info = linalg_dposv(n, Kchol, gpsep.Ki)
    if info:
        print("d = ", gpsep.d)
        raise ValueError(f"Bad Cholesky decomposition (info={info}), g={gpsep.g}")
    gpsep.ldetK = log_determinant_chol(Kchol)
    del Kchol

    # phi <- t(Z) %*% Ki %*% Z
    gpsep.KiZ = None
    gpsep = calc_ZtKiZ_sep(gpsep)

    # Calculate derivatives ?
    gpsep.dK = None
    if dK:
        gpsep = newdKGPsep(gpsep)

    # Return new structure
    return gpsep


def newGPsep(X, Z, d, g, dK=False) -> GPsep:
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


def predGPsep(gpsep, nn, XX, nonug, mean, Sigma, df, llik) -> tuple:
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

    Returns:
        tuple: A tuple containing the mean, Sigma, df, and llik.
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


def pred_generic(phidf, Z, Ki, k, mean, Sigma) -> tuple:
    """
    Generic function for GP prediction.

    Args:
        phidf (float): The phi/df value.
        Z (ndarray): The response vector.
        Ki (ndarray): The inverse covariance matrix.
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


def predictGPsep(gpsep, XX, lite=False, nonug=False) -> dict:
    """
    Obtain the parameters to a multivariate-t distribution describing the predictive surface
    of the fitted GP model.

    Args:
        gpsep (GPsep): The GPsep object.
        XX (ndarray): The predictive locations.
        lite (bool): Flag to indicate whether to compute only the diagonal of Sigma.
        nonug (bool): Flag to indicate whether to use nugget.

    Returns:
        dict: A dictionary containing the mean, Sigma (or s2), df, and llik.

    Examples:
        >>> import numpy as np
        >>> from spotpython.gp.gp_sep import newGPsep, predictGPsep
        >>> from spotpython.gp.functions import f2d
        >>> import matplotlib.pyplot as plt
        >>> # Design with N=441
        >>> x = np.linspace(-2, 2, 11)
        >>> X = np.array(np.meshgrid(x, x)).T.reshape(-1, 2)
        >>> Z = f2d(X)
        >>> # Fit a GP
        >>> gpsep = newGPsep(X, Z, d=0.35, g=1/1000)
        >>> # Predictive grid with NN=400
        >>> xx = np.linspace(-1.9, 1.9, 20)
        >>> XX = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2)
        >>> ZZ = f2d(XX)
        >>> # Predict
        >>> p = predictGPsep(gpsep, XX)
        >>> # RMSE: compare to similar experiment in aGP docs
        >>> rmse = np.sqrt(np.mean((p["mean"] - ZZ) ** 2))
        >>> print("RMSE:", rmse)
        >>> # Visualize the result
        >>> plt.figure(figsize=(12, 6))
        >>> plt.subplot(1, 2, 1)
        >>> plt.imshow(p["mean"].reshape(len(xx), len(xx)), extent=(xx.min(), xx.max(), xx.min(), xx.max()), origin='lower', cmap='hot')
        >>> plt.colorbar()
        >>> plt.xlabel("x1")
        >>> plt.ylabel("x2")
        >>> plt.title("Predictive Mean")
        >>> plt.subplot(1, 2, 2)
        >>> plt.imshow((p["mean"] - ZZ).reshape(len(xx), len(xx)), extent=(xx.min(), xx.max(), xx.min(), xx.max()), origin='lower', cmap='hot')
        >>> plt.colorbar()
        >>> plt.xlabel("x1")
        >>> plt.ylabel("x2")
        >>> plt.title("Residuals")
        >>> plt.show()
    """
    nn = XX.shape[0]
    m = XX.shape[1]
    if nn == 0:
        raise ValueError("XX bad dims")

    if lite:
        # Lite means does not compute full Sigma, only diag
        mean_out = np.zeros(nn)
        s2_out = np.zeros(nn)
        df_out = np.zeros(1)
        llik_out = np.zeros(1)

        mean_out, s2_out, df_out, llik_out = predGPsep_lite(gpsep, m, nn, XX, lite_in=True, nonug_in=nonug, mean_out=mean_out, Sigma_out=s2_out, df_out=df_out, llik_out=llik_out)

        # Return parameterization
        return {"mean": mean_out, "s2": s2_out, "df": df_out, "llik": llik_out}

    else:
        # Compute full predictive covariance matrix
        mean_out = np.zeros(nn)
        Sigma_out = np.zeros(nn * nn)
        df_out = np.zeros(1)
        llik_out = np.zeros(1)

        if m != gpsep.m:
            raise ValueError(f"ncol(X)={m} does not match GPsep/C-side ({gpsep.m})")

        # Sanity check and XX representation
        XX = XX.reshape(nn, m)
        Sigma = Sigma_out.reshape(nn, nn)

        mean_out, Sigma, df_out, llik_out = predGPsep(gpsep, nn, XX, nonug, mean_out, Sigma, df_out, llik_out)

        # Coerce matrix output
        Sigma = Sigma_out.reshape(nn, nn)

        # Return parameterization
        return {"mean": mean_out, "Sigma": Sigma, "df": df_out, "llik": llik_out}
