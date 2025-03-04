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

    def newdK(self):
        """
        Allocate new space for dK calculations, and calculate derivatives.
        Updates the dK and dk fields of the GPsep object.

        Returns:
            GPsep: The updated GPsep object.
        """
        assert self.dK is None
        self.dK = np.empty((self.m, self.n, self.n))
        for j in range(self.m):
            self.dK[j] = new_matrix(self.n, self.n)
        self.dk = diff_covar_sep_symm(self.m, self.X, self.n, self.d, self.K, self.dK)
        return self

    def calc_ZtKiZ(self):
        """
        Re-calculates phi = ZtKiZ from Ki and Z stored in the GP object; also update KiZ on which it depends.

        Returns:
            GPsep: The updated GPsep object.
        """
        assert self is not None
        if self.KiZ is None:
            self.KiZ = new_vector(self.n)

        # Ensure Z is a column vector
        Z = self.Z
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        # Calculate phi = t(Z) %*% Ki %*% Z
        KiZ = np.dot(self.Ki, Z)
        phi = np.dot(Z.T, KiZ)

        # Since phi is a 1x1 matrix, we extract the scalar value
        phi = phi[0, 0]

        self.phi = phi
        self.KiZ = KiZ
        return self

    def build(self, dK):
        """
        Intended for newly created separable GPs, e.g., via newGPsep.
        Does all of the correlation calculations, etc., after data and parameters are defined.
        Similar to buildGP except calculates gradient dK.

        Args:
            dK (int): Flag to indicate whether to calculate derivatives.

        Returns:
            GPsep: The updated GPsep object.
        """
        assert self is not None and self.K is None
        n = self.n
        m = self.m
        X = self.X

        # Build covariance matrix
        self.K = new_matrix(n, n)
        self.K = covar_sep_symm(m, X, n, self.d, self.g, self.K)

        # Invert covariance matrix
        self.Ki = new_id_matrix(n)
        Kchol = new_dup_matrix(self.K, n, n)
        self.Ki, info = linalg_dposv(n, Kchol, self.Ki)
        if info:
            print("d = ", self.d)
            raise ValueError(f"Bad Cholesky decomposition (info={info}), g={self.g}")
        self.ldetK = log_determinant_chol(Kchol)
        del Kchol

        # phi <- t(Z) %*% Ki %*% Z
        self.KiZ = None
        self = self.calc_ZtKiZ()

        # Calculate derivatives ?
        self.dK = None
        if dK:
            self = self.newdK()
        return self

    def predict(self, XX, lite=False, nonug=False):
        """
        Obtain the parameters to a multivariate-t distribution describing the predictive surface
        of the fitted GP model.

        Args:
            XX (ndarray): The predictive locations.
            lite (bool): Flag to indicate whether to compute only the diagonal of Sigma.
            nonug (bool): Flag to indicate whether to use nugget.

        Returns:
            dict: A dictionary containing the mean, Sigma (or s2), df, and llik.

        Examples:
            >>> import numpy as np
            >>> from spotpython.gp.gp_sep import GPsep, newGPsep
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> Z = np.array([1.0, 2.0, 3.0])
            >>> d = 1.0
            >>> g = 0.1
            >>> gpsep = newGPsep(X, Z, d, g, dK=False)
            >>> XX = np.array([[1, 2], [3, 4]])
            >>> result = gpsep.predict(XX, lite=False, nonug=False)
            >>> print(result)
            {'mean': array([1., 2.]), 'Sigma': array([[1., 1.], [1., 1.]]), 'df': array([3.]), 'llik': array([-3.465
            735902799726])}
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

            mean_out, s2_out, df_out, llik_out = predGPsep_lite(self, m, nn, XX, lite_in=True, nonug_in=nonug, mean_out=mean_out, Sigma_out=s2_out, df_out=df_out, llik_out=llik_out)
            return {"mean": mean_out, "s2": s2_out, "df": df_out, "llik": llik_out}

        else:
            # Compute full predictive covariance matrix
            mean_out = np.zeros(nn)
            Sigma_out = np.zeros(nn * nn)
            df_out = np.zeros(1)
            llik_out = np.zeros(1)

            if m != self.m:
                raise ValueError(f"ncol(X)={m} does not match GPsep/C-side ({self.m})")

            XX = XX.reshape(nn, m)
            Sigma = Sigma_out.reshape(nn, nn)
            mean_out, Sigma, df_out, llik_out = self.pred(nn, XX, nonug, mean_out, Sigma, df_out, llik_out)
            Sigma = Sigma_out.reshape(nn, nn)
            return {"mean": mean_out, "Sigma": Sigma, "df": df_out, "llik": llik_out}

    def pred(self, nn, XX, nonug, mean, Sigma, df, llik):
        """
        Return the student-t predictive equations,
        i.e., parameters to a multivariate t-distribution
        for XX predictive locations of dimension (n*m).

        Args:
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
        n = self.n
        m = self.m

        # Are we using a nugget in the final calculation
        if nonug:
            g = np.finfo(float).eps
        else:
            g = self.g

        # Variance (s2) components
        df[0] = float(n)
        phidf = self.phi / df[0]
        # Calculate marginal likelihood (since we have the bits)
        llik[0] = -0.5 * (df[0] * np.log(0.5 * self.phi) + self.ldetK)

        # k <- covar(X1=X, X2=XX, d=Zt$d, g=0)
        k = covar_sep(m, self.X, n, XX, nn, self.d, 0.0)
        # Sigma <- covar(X1=XX, d=Zt$d, g=Zt$g)
        Sigma[:] = covar_sep_symm(m, XX, nn, self.d, g)
        # Call generic function that would work for all GP covariance specs
        mean, Sigma = self.pred_generic(phidf, self.Z, self.Ki, k, mean, Sigma)
        return mean, Sigma, df, llik

    def pred_generic(self, phidf, Z, Ki, k, mean, Sigma):
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
    return gpsep.build(dK)
