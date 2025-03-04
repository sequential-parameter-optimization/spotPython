import numpy as np
from spotpython.gp.linalg import linalg_dposv
from spotpython.gp.covar import covar_sep_symm, covar_sep, diff_covar_sep_symm
from spotpython.gp.util import log_determinant_chol
from spotpython.gp.matrix import new_matrix, new_vector, new_id_matrix, new_dup_matrix, new_dup_vector


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
    gpsep.Ki, info = linalg_dposv(n, Kchol, gpsep.Ki)
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
