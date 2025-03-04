import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv
from spotpython.gp.likelihood import nlsep, gradnlsep
from spotpython.gp.distances import covar_anisotropic


class GPRegressor:
    def __init__(self):
        self.eps = np.sqrt(np.finfo(float).eps)
        self.fitted = False

    def fit(self, X, y) -> "GPRegressor":
        """
        Fit the Gaussian Process model.

        Args:
            X (np.ndarray): Training input matrix of shape (n, m).
            y (np.ndarray): Training response vector of shape (n,).

        Returns:
            self: Fitted model.
        """
        self.n, self.m = X.shape

        # Optimize parameters
        outg = minimize(
            lambda par: nlsep(par, X, y),
            x0=np.concatenate([np.repeat(0.1, self.m), [0.1 * np.var(y)]]),
            jac=lambda par: gradnlsep(par, X, y),
            method="L-BFGS-B",
            bounds=[(self.eps, 10)] * self.m + [(self.eps, np.var(y))],
        )

        # Compute covariance matrices
        K = covar_anisotropic(X, d=outg.x[: self.m], g=outg.x[self.m])
        Ki = inv(K)
        tau2hat = (y.T @ Ki @ y) / len(X)

        self.X = X
        self.y = y
        self.outg = outg
        self.Ki = Ki
        self.tau2hat = tau2hat
        self.fitted = True

        return self

    def predict(self, XX) -> tuple:
        """
        Predict using the Gaussian Process model.

        Args:
            XX (np.ndarray): Test input matrix of shape (n_test, m).

        Returns:
            tuple: Predicted mean (mup2) and covariance (Sigmap2).

        Raises:
            RuntimeError: If the model is not fitted.

        Examples:
            >>> import numpy as np
            >>> from spotpython.gp.regressor import GPRegressor
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> y = np.array([1, 2, 3])
            >>> XX = np.array([[1, 2], [3, 4]])
            >>> gp_model = GPRegressor()
            >>> gp_model.fit(X, y)
            >>> mup2, Sigmap2 = gp_model.predict(XX)
            >>> print(mup2)
            [1. 2.]
            >>> print(Sigmap2)
            [[1. 1.]
             [1. 1.]]

        """
        if not self.fitted:
            raise RuntimeError("The model must be fitted before calling predict.")

        KXX = covar_anisotropic(XX, d=self.outg.x[: self.m], g=self.outg.x[self.m])
        KX = covar_anisotropic(XX, self.X, d=self.outg.x[: self.m], g=0.0)
        mup2 = KX @ self.Ki @ self.y
        Sigmap2 = self.tau2hat * (KXX - KX @ self.Ki @ KX.T)

        return mup2, Sigmap2
