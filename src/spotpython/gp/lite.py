import numpy as np
from spotpython.gp.covar import covar_sep


def new_predutilGPsep_lite(gpsep, nn, XX) -> tuple:
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


def predGPsep_lite(gpsep, m, nn, XX, lite_in, nonug_in, mean_out, Sigma_out, df_out, llik_out) -> tuple:
    """
    Perform a lite prediction using the GPsep model.

    Args:
        gpsep (GPsep): The GPsep object.
        m (int): The number of input dimensions.
        nn (int): The number of predictive locations.
        XX (ndarray): The predictive locations.
        lite_in (bool): Flag to indicate whether to use lite prediction.
        nonug_in (bool): Flag to indicate whether to use nugget.
        mean_out (ndarray): The output mean.
        Sigma_out (ndarray): The output covariance matrix.
        df_out (ndarray): The output degrees of freedom.
        llik_out (ndarray): The output log-likelihood.

    Returns:
        tuple: A tuple containing the mean, Sigma (or s2), df, and llik.
    """
    # Sanity checks
    assert df_out is not None
    df_out[0] = gpsep.n

    # Are we using a nugget in the final calculation
    if nonug_in:
        g = np.finfo(float).eps
    else:
        g = gpsep.g

    # Utility calculations
    k, ktKi, ktKik = new_predutilGPsep_lite(gpsep, nn, XX)

    # mean <- ktKi %*% Z
    if mean_out is not None:
        mean_out[:] = np.dot(ktKi, gpsep.Z)

    # Sigma <- phi*(Sigma - ktKik)/df
    if Sigma_out is not None:
        phidf = gpsep.phi / df_out[0]
        for i in range(nn):
            Sigma_out[i] = phidf * (1.0 + g - ktKik[i])

    # Calculate marginal likelihood (since we have the bits)
    if llik_out is not None:
        llik_out[0] = -0.5 * (gpsep.n * np.log(0.5 * gpsep.phi) + gpsep.ldetK)

    return mean_out, Sigma_out, df_out, llik_out


def new_predutil_generic_lite(n, Ki, nn, k) -> tuple:
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
