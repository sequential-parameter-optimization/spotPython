import numpy as np
from spotpython.gp.linalg import linalg_dposv
from spotpython.gp.covar import covar_sep_symm, covar_sep, diff_covar_sep_symm
from spotpython.gp.util import log_determinant_chol
from spotpython.gp.matrix import new_vector, new_id_matrix, new_dup_matrix
from spotpython.gp.lite import predGPsep_lite


class GPsep:
    """A class to represent a Gaussian Process with separable covariance."""

    def __init__(self, m: int = None, n: int = None, X: np.ndarray = None, Z: np.ndarray = None, d: np.ndarray = None, g: float = None) -> None:
        """
        Initialize the GP model with data and hyperparameters.

        Args:
            m (int): Number of input dimensions.
            n (int): Number of observations.
            X (np.ndarray): Input data matrix of shape (n, m).
            Z (np.ndarray): Output data vector of length n.
            d (np.ndarray): Length-scale parameters.
            g (float): Nugget parameter.
        """
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

    def fit(self, X: np.ndarray, Z: np.ndarray, d: np.ndarray, g: float, dK: bool = False) -> "GPsep":
        """
        Fits the GP model with training data.

        Args:
            X (np.ndarray): The input data matrix of shape (n, m).
            Z (np.ndarray): The output data vector of length n.
            d (Union[np.ndarray, float]): The length-scale parameters.
            g (float): The nugget parameter.
            dK (bool): Flag to indicate whether to calculate derivatives.

        Returns:
            GPsep: The fitted GPsep object.
        """
        n, m = X.shape
        if n == 0:
            raise ValueError("X must be a matrix with rows.")
        if len(Z) != n:
            raise ValueError(f"X has {n} rows but Z length is {len(Z)}")

        self.m = m
        self.n = n
        self.X = X
        self.Z = Z
        self.d = np.full(m, d) if isinstance(d, (int, float)) else d
        if len(self.d) != m:
            raise ValueError(f"Length of d ({len(self.d)}) does not match ncol(X) ({m})")
        self.g = g

        self.build(dK)
        return self

    def newdK(self) -> None:
        """
        Allocate space for derivative calculations and compute them.
        """
        if self.dK is not None:
            raise RuntimeError("dK calculations have already been initialized.")

        self.dK = diff_covar_sep_symm(self.m, self.X, self.n, self.d, self.K)

    def calc_ZtKiZ(self) -> None:
        """
        Recalculate phi and related components from Ki and Z.
        """
        if self.KiZ is None:
            self.KiZ = new_vector(self.n)

        Z = self.Z.reshape(-1, 1)
        KiZ = np.dot(self.Ki, Z)
        phi = np.dot(Z.T, KiZ)
        self.phi = phi[0, 0]
        self.KiZ = KiZ

    def build(self, dK: bool) -> None:
        """
        Completes all correlation calculations after data is defined.

        Args:
            dK (bool): Flag to indicate whether to calculate derivatives.
        """
        if self.K is not None:
            raise RuntimeError("Covariance matrix has already been built.")

        self.K = covar_sep_symm(self.m, self.X, self.n, self.d, self.g)
        self.Ki = new_id_matrix(self.n)
        Kchol = new_dup_matrix(self.K, self.n, self.n)
        self.Ki, info = linalg_dposv(self.n, Kchol, self.Ki)
        if info != 0:
            raise ValueError(f"Cholesky decomposition failed (info={info}) with g={self.g}.")
        self.ldetK = log_determinant_chol(Kchol)

        self.calc_ZtKiZ()
        if dK:
            self.newdK()

    def predict(self, XX: np.ndarray, lite: bool = False, nonug: bool = False) -> dict:
        """
        Predict the Gaussian Process output at new input points.

        Args:
            XX (np.ndarray): The predictive locations.
            lite (bool): Flag to indicate whether to compute only the diagonal of Sigma.
            nonug (bool): Flag to indicate whether to use nugget.

        Returns:
            dict: A dictionary containing the mean, Sigma (or s2), df, and llik.
        """
        if lite:
            return self._predict_lite(XX, nonug)
        else:
            return self._predict_full(XX, nonug)

    def _predict_lite(self, XX: np.ndarray, nonug: bool) -> dict:
        """
        Predict only the diagonal of Sigma—optimized for speed.

        Args:
            XX (np.ndarray): The predictive locations.
            nonug (bool): Flag to indicate whether to use nugget.

        Returns:
            dict: A dictionary containing the mean, s2, df, and llik.
        """
        nn = XX.shape[0]
        m = XX.shape[1]
        mean_out, s2_out, df_out, llik_out = predGPsep_lite(self, m, nn, XX, lite_in=True, nonug_in=nonug)
        return {"mean": mean_out, "s2": s2_out, "df": df_out, "llik": llik_out}

    def _predict_full(self, XX: np.ndarray, nonug: bool) -> dict:
        """
        Compute full predictive covariance matrix.

        Args:
            XX (np.ndarray): The predictive locations.
            nonug (bool): Flag to indicate whether to use nugget.

        Returns:
            dict: A dictionary containing the mean, Sigma, df, and llik.
        """
        nn, m = XX.shape
        if m != self.m:
            raise ValueError(f"ncol(X)={m} does not match GPsep model ({self.m})")

        mean_out = np.zeros(nn)
        Sigma_out = np.zeros((nn, nn))
        df_out = np.zeros(1)
        llik_out = np.zeros(1)

        mean_out, Sigma_out, df_out, llik_out = self.pred(nn, XX, nonug, mean_out, Sigma_out, df_out, llik_out)
        return {"mean": mean_out, "Sigma": Sigma_out, "df": df_out, "llik": llik_out}

    def pred(self, nn: int, XX: np.ndarray, nonug: bool, mean: np.ndarray, Sigma: np.ndarray, df: np.ndarray, llik: np.ndarray) -> tuple:
        """
        Return the predictive mean and covariance.

        Args:
            nn (int): Number of predictive locations.
            XX (np.ndarray): The predictive locations.
            nonug (bool): Flag to indicate whether to use nugget.
            mean (np.ndarray): The output mean.
            Sigma (np.ndarray): The output covariance matrix.
            df (np.ndarray): The output degrees of freedom.
            llik (np.ndarray): The output log-likelihood.

        Returns:
            tuple: A tuple containing the mean, Sigma, df, and llik.
        """
        n = self.n
        g = np.finfo(float).eps if nonug else self.g
        df[0] = float(n)
        phidf = self.phi / df[0]
        llik[0] = -0.5 * (df[0] * np.log(0.5 * self.phi) + self.ldetK)
        k = covar_sep(self.m, self.X, n, XX, nn, self.d, 0.0)
        Sigma[...] = covar_sep_symm(self.m, XX, nn, self.d, g)
        mean, Sigma = self.pred_generic(phidf, self.Z, self.Ki, k, mean, Sigma)
        return mean, Sigma, df, llik

    def pred_generic(self, phidf: float, Z: np.ndarray, Ki: np.ndarray, k: np.ndarray, mean: np.ndarray, Sigma: np.ndarray) -> tuple:
        """
        Generic GP prediction calculation.

        Args:
            phidf (float): The phi/df value.
            Z (np.ndarray): The response vector.
            Ki (np.ndarray): The inverse covariance matrix.
            k (np.ndarray): The covariance matrix between training and predictive locations.
            mean (np.ndarray): The output mean.
            Sigma (np.ndarray): The output covariance matrix.

        Returns:
            tuple: A tuple containing the mean and Sigma.
        """
        ktKi = np.dot(k.T, Ki)
        mean[:] = np.dot(ktKi, Z).reshape(-1)
        Sigma[...] = phidf * (Sigma - np.dot(ktKi, k))
        return mean, Sigma


def newGPsep(X: np.ndarray, Z: np.ndarray, d: float, g: float, dK: bool = False) -> GPsep:
    """
    Instantiate a new GPsep model.

    Args:
        X (np.ndarray): The input data matrix of shape (n, m).
        Z (np.ndarray): The output data vector of length n.
        d (float): The length-scale parameter.
        g (float): The nugget parameter.
        dK (bool): Flag to indicate whether to calculate derivatives.

    Returns:
        GPsep: The newly created GPsep object.
    """
    gpsep = GPsep()
    return gpsep.fit(X, Z, d, g, dK)
