import numpy as np
from spotpython.gp.covar import covar_sep


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
