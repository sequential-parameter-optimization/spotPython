import numpy as np
from numpy.linalg import LinAlgError, cond
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import erf
import matplotlib.pyplot as plt
from numpy import linspace, append
from scipy.spatial.distance import cdist, pdist, squareform
from spotpython.surrogate.plot import plotkd

# --- The New Kriging Class with Nyström Approximation as introduced in v0.34.0 ---


class Kriging(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible Kriging model class for regression tasks,
    extended with an optional Nyström approximation for scaling.

    Public API and core behavior match src/spotpython/surrogate/kriging.py.
    The same basis function/correlation definitions are used. When use_nystrom=True,
    training and prediction use low-rank Woodbury solves based on m inducing points.

    Attributes:
        eps (float): A small regularization term to reduce ill-conditioning.
        penalty (float): The penalty value used if the correlation matrix is ill-conditioned.
        logtheta_loglambda_p_ (np.ndarray): Best-fit log(theta), log(lambda), and p parameters from fit().
        U_ (np.ndarray): Cholesky factor of the correlation matrix (exact mode).
        X_ (np.ndarray): Training input data (n x k).
        y_ (np.ndarray): Training target values (n,).
        negLnLike (float): Negative log-likelihood at the optimum.
        Psi_ (np.ndarray): Correlation matrix used in likelihood (exact mode).
        method (str): "interpolation", "regression", or "reinterpolation".
        isotropic (bool): If True, one theta for all ordered dims.
        use_nystrom (bool): If True, use Nyström low-rank solves.
        nystrom_m (int): Number of inducing points (landmarks).
        nystrom_seed (int): RNG seed for landmark selection.

    Notes (Forrester et al., Engineering Design via Surrogate Modelling, Ch. 2/3/6):
        - Correlation: R = exp(-D), with D weighted distances (Ch. 2).
        - Ordinary Kriging μ, σ^2, concentrated likelihood (Ch. 3).
        - Prediction f̂, variance s^2, and Expected Improvement (Ch. 3 & 6).
    """

    def __init__(
        self,
        eps: float = None,
        penalty: float = 1e4,
        method: str = "regression",
        var_type: List[str] = ["num"],
        name: str = "Kriging",
        seed: int = 124,
        model_optimizer=None,
        model_fun_evals: Optional[int] = None,
        n_theta: Optional[int] = None,
        min_theta: float = -3.0,
        max_theta: float = 2.0,
        theta_init_zero: bool = False,
        p_val: float = 2.0,
        n_p: int = 1,
        optim_p: bool = False,
        min_p: float = 1.0,
        max_p: float = 2.0,
        min_Lambda: float = -9.0,
        max_Lambda: float = 0.0,
        log_level: int = 50,
        spot_writer=None,
        counter=None,
        metric_factorial: str = "canberra",
        isotropic: bool = False,
        theta: Optional[np.ndarray] = None,
        Lambda: Optional[float] = None,
        # Nyström options
        use_nystrom: bool = False,
        nystrom_m: Optional[int] = None,
        nystrom_seed: int = 1234,
        **kwargs,
    ):
        if eps is None:
            self.eps = self._get_eps()
        else:
            if eps <= 0:
                raise ValueError("eps must be positive")
            self.eps = eps
        self.penalty = penalty
        self.var_type = var_type
        self.name = name
        self.seed = seed
        self.log_level = log_level
        self.spot_writer = spot_writer
        self.counter = counter
        self.metric_factorial = metric_factorial
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_Lambda = min_Lambda
        self.max_Lambda = max_Lambda
        self.min_p = min_p
        self.max_p = max_p
        self.n_theta = n_theta  # Will be set during fit
        self.isotropic = isotropic
        self.p_val = p_val
        self.n_p = n_p
        self.optim_p = optim_p
        self.theta_init_zero = theta_init_zero
        self.theta = theta
        self.Lambda = Lambda
        self.model_optimizer = model_optimizer or differential_evolution
        self.model_fun_evals = model_fun_evals or 100

        # Logging information
        self.log = {}
        self.log["negLnLike"] = []
        self.log["theta"] = []
        self.log["p"] = []
        self.log["Lambda"] = []

        self.logtheta_loglambda_p_ = None
        self.U_ = None
        self.X_ = None
        self.y_ = None
        self.negLnLike = None
        self.Psi_ = None
        if method not in ["interpolation", "regression", "reinterpolation"]:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation']")
        self.method = method
        self.return_ei = False
        self.return_std = False

        # Variable-type masks (set during fit)
        self.num_mask = None
        self.factor_mask = None
        self.int_mask = None
        self.ordered_mask = None

        # Nyström-specific attributes
        self.use_nystrom = use_nystrom
        self.nystrom_m = nystrom_m
        self.nystrom_seed = nystrom_seed
        self.landmark_idx_ = None
        self.X_m_ = None  # landmarks (m x k)
        self.W_chol_ = None  # chol(W) of K_mm
        self.M_chol_ = None  # chol(M) where M = W + (1/λ) C^T C
        self.C_ = None  # K_nm
        self.lambda_lin_ = None  # λ (linear scale)
        self.mu_ = None  # μ̂
        self.sigma2_ = None  # σ̂^2
        self.Rinv_one_ = None  # R^{-1} 1 (n,)
        self.Rinv_r_ = None  # R^{-1} r (n,)

    def _get_eps(self) -> float:
        eps = np.finfo(float).eps
        return np.sqrt(eps)

    def _set_variable_types(self) -> None:
        if len(self.var_type) < self.k:
            self.var_type = ["num"] * self.k
        var_type_array = np.array(self.var_type)
        self.num_mask = var_type_array == "num"
        self.factor_mask = var_type_array == "factor"
        self.int_mask = var_type_array == "int"
        self.ordered_mask = np.isin(var_type_array, ["int", "num", "float"])

    def get_model_params(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "k": self.k,
            "logtheta_loglambda_p_": self.logtheta_loglambda_p_,
            "U": self.U_,
            "X": self.X_,
            "y": self.y_,
            "negLnLike": self.negLnLike,
            "inf_Psi": getattr(self, "inf_Psi", None),
            "cnd_Psi": getattr(self, "cnd_Psi", None),
        }

    def _update_log(self) -> None:
        self.log["negLnLike"] = append(self.log["negLnLike"], self.negLnLike)
        self.log["theta"] = append(self.log["theta"], self.theta)
        if self.optim_p:
            self.log["p"] = append(self.log["p"], self.p_val)
        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.log["Lambda"] = append(self.log["Lambda"], self.Lambda)
        self.log_length = len(self.log["negLnLike"])
        if self.spot_writer is not None:
            negLnLike = float(self.negLnLike)
            self.spot_writer.add_scalar("spot_negLnLike", negLnLike, (self.counter or 0) + self.log_length)
            theta = np.array(self.theta, copy=True)
            self.spot_writer.add_scalars("spot_theta", {f"theta_{i}": theta[i] for i in range(self.n_theta)}, (self.counter or 0) + self.log_length)
            if (self.method == "regression") or (self.method == "reinterpolation"):
                Lambda = np.array(self.Lambda, copy=True)
                self.spot_writer.add_scalar("spot_Lambda", float(Lambda), (self.counter or 0) + self.log_length)
            if self.optim_p:
                p = np.array(self.p_val, copy=True)
                self.spot_writer.add_scalars("spot_p", {f"p_{i}": p[i] for i in range(self.n_p)}, (self.counter or 0) + self.log_length)
            self.spot_writer.flush()

    def _set_theta(self) -> None:
        self.n_theta = 1 if self.isotropic else self.k

    def _get_theta10_from_logtheta(self) -> np.ndarray:
        theta10 = np.power(10.0, self.theta)
        if self.n_theta == 1:
            theta10 = theta10 * np.ones(self.k)
        return theta10

    def _reshape_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, self.k)
        else:
            if X.shape[1] != self.k:
                if X.shape[0] == self.k:
                    X = X.T
                elif self.k == 1:
                    X = X.reshape(-1, 1)
                else:
                    raise ValueError(f"X has shape {X.shape}, expected (*, {self.k}).")
        return X

    # -------- Basis correlation construction (identical to kriging.py) --------

    def build_Psi(self) -> np.ndarray:
        """
        Constructs the correlation matrix upper triangle Psi (R = exp(-D)), see Forrester Ch. 2.
        """
        try:
            n, k = self.X_.shape
            theta10 = self._get_theta10_from_logtheta()

            Psi = np.zeros((n, n), dtype=np.float64)

            if self.ordered_mask.any():
                X_ordered = self.X_[:, self.ordered_mask]
                D_ordered = squareform(pdist(X_ordered, metric="sqeuclidean", w=theta10[self.ordered_mask]))
                Psi += D_ordered

            if self.factor_mask.any():
                X_factor = self.X_[:, self.factor_mask]
                D_factor = squareform(pdist(X_factor, metric=self.metric_factorial, w=theta10[self.factor_mask]))
                Psi += D_factor

            Psi = np.exp(-Psi)

            self.inf_Psi = np.isinf(Psi).any()
            self.cnd_Psi = cond(Psi)

            return np.triu(Psi, k=1)
        except LinAlgError as err:
            print("Building Psi failed. Error: %s, Type: %s", err, type(err))
            raise

    def _kernel_cross(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Cross-correlation matrix K(A,B) using the same distance definition as build_Psi/build_psi_vec.
        Returns exp(-D) with D composed from ordered and factor contributions.
        """
        A = np.asarray(A)
        B = np.asarray(B)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if B.ndim == 1:
            B = B.reshape(1, -1)
        theta10 = self._get_theta10_from_logtheta()

        D = np.zeros((A.shape[0], B.shape[0]), dtype=float)
        if self.ordered_mask.any():
            Ao = A[:, self.ordered_mask]
            Bo = B[:, self.ordered_mask]
            D += cdist(Ao, Bo, metric="sqeuclidean", w=theta10[self.ordered_mask])
        if self.factor_mask.any():
            Af = A[:, self.factor_mask]
            Bf = B[:, self.factor_mask]
            D += cdist(Af, Bf, metric=self.metric_factorial, w=theta10[self.factor_mask])

        return np.exp(-D)

    def build_psi_vec(self, x: np.ndarray) -> np.ndarray:
        """
        ψ(x) = K(x, X) with the same metric as build_Psi, Forrester Ch. 2.
        """
        try:
            psi = self._kernel_cross(np.asarray(x), self.X_).ravel()
            return psi
        except np.linalg.LinAlgError as err:
            print("Building psi failed due to a linear algebra error: %s. Error type: %s", err, type(err))
            return np.zeros(self.X_.shape[0])

    # -------- Exact (Cholesky) likelihood --------

    def _likelihood_exact(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        X = self.X_
        y = self.y_.flatten()
        self.theta = x[: self.n_theta]

        if (self.method == "regression") or (self.method == "reinterpolation"):
            lambda_ = 10.0 ** x[self.n_theta : self.n_theta + 1]
            if self.optim_p:
                self.p_val = x[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            lambda_ = self.eps
            if self.optim_p:
                self.p_val = x[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        n = X.shape[0]
        one = np.ones(n)

        Psi_up = self.build_Psi()
        Psi = Psi_up + Psi_up.T + np.eye(n) + np.eye(n) * lambda_

        try:
            U = np.linalg.cholesky(Psi)
        except LinAlgError:
            return self.penalty, Psi, None

        LnDetPsi = 2.0 * np.sum(np.log(np.abs(np.diag(U))))

        temp_y = np.linalg.solve(U, y)
        temp_one = np.linalg.solve(U, one)
        vy = np.linalg.solve(U.T, temp_y)
        vone = np.linalg.solve(U.T, temp_one)

        mu = (one @ vy) / (one @ vone)

        resid = y - one * mu
        tresid = np.linalg.solve(U, resid)
        tresid = np.linalg.solve(U.T, tresid)
        SigmaSqr = (resid @ tresid) / n

        negLnLike = (n / 2.0) * np.log(SigmaSqr) + 0.5 * LnDetPsi
        return negLnLike, Psi, U

    # -------- Nyström likelihood (low-rank) --------

    def _nystrom_setup(self) -> None:
        """
        Selects landmarks and builds K_mm (W) and K_nm (C) for the current theta.
        """
        n = self.n
        m = self.nystrom_m or max(10, min(n // 2, 300))
        m = min(m, n - 1) if n > 1 else 1
        rng = np.random.default_rng(self.nystrom_seed)
        self.landmark_idx_ = rng.choice(n, size=m, replace=False)
        self.X_m_ = self.X_[self.landmark_idx_]

        # W = K(X_m, X_m), C = K(X, X_m)
        W = self._kernel_cross(self.X_m_, self.X_m_)
        # Force symmetry and unit-diagonal structure exactly as full K would have
        W = (W + W.T) / 2.0
        np.fill_diagonal(W, 1.0)
        C = self._kernel_cross(self.X_, self.X_m_)
        self.C_ = C

        # Precompute chol(W) for logdet and stability checks
        try:
            self.W_chol_ = np.linalg.cholesky(W)
        except LinAlgError:
            # Fall back to adding tiny jitter and retry
            W_jit = W + 1e-12 * np.eye(W.shape[0])
            self.W_chol_ = np.linalg.cholesky(W_jit)

    def _woodbury_solve(self, v: np.ndarray) -> np.ndarray:
        """
        Applies R^{-1} v using Woodbury:
          R = λ I + C W^{-1} C^T  has inverse
          R^{-1} = (1/λ) I − (1/λ^2) C (W + (1/λ) C^T C)^{-1} C^T.
        Uses precomputed self.M_chol_ (chol of W + (1/λ) C^T C).
        """
        lam = float(self.lambda_lin_)
        C = self.C_
        # rhs = C^T v
        rhs = C.T @ v
        # Solve α from M α = rhs
        alpha = np.linalg.solve(self.M_chol_, np.linalg.solve(self.M_chol_.T, rhs))
        # R^{-1} v
        return (v / lam) - (C @ alpha) / (lam * lam)

    def _likelihood_nystrom(self, x: np.ndarray) -> Tuple[float, None, None]:
        """
        Low-rank (Nyström) concentrated likelihood using matrix determinant lemma and Woodbury.
        Returns (negLnLike, None, None) to match the exact signature.
        """
        X = self.X_
        y = self.y_.flatten()
        self.theta = x[: self.n_theta]

        if (self.method == "regression") or (self.method == "reinterpolation"):
            lambda_lin = 10.0 ** x[self.n_theta : self.n_theta + 1]
            lambda_lin = float(lambda_lin)
            if self.optim_p:
                self.p_val = x[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            lambda_lin = float(self.eps)
            if self.optim_p:
                self.p_val = x[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        self.lambda_lin_ = lambda_lin

        # Build Nyström structures for current theta
        self._nystrom_setup()
        C = self.C_
        W_chol = self.W_chol_
        W = W_chol @ W_chol.T

        n = X.shape[0]
        m = W.shape[0]
        one = np.ones(n)

        # M = W + (1/λ) C^T C
        CtC = C.T @ C
        M = W + (CtC / lambda_lin)

        # Cholesky of M
        try:
            self.M_chol_ = np.linalg.cholesky(M)
        except LinAlgError:
            # add jitter
            M_jit = M + 1e-12 * np.eye(m)
            self.M_chol_ = np.linalg.cholesky(M_jit)

        # log|R| via determinant lemma:
        # |R| = λ^n det(I_m + (1/λ) W^{-1} C^T C) = λ^n det(W^{-1} M)
        # log|R| = n log λ + log|M| − log|W|
        logdetW = 2.0 * np.sum(np.log(np.diag(W_chol)))
        logdetM = 2.0 * np.sum(np.log(np.diag(self.M_chol_)))
        LnDetR = n * np.log(lambda_lin) + (logdetM - logdetW)

        # R^{-1} y and R^{-1} 1 via Woodbury
        Rinv_y = self._woodbury_solve(y)
        Rinv_one = self._woodbury_solve(one)

        # μ̂ and σ̂^2
        mu = (one @ Rinv_y) / (one @ Rinv_one)
        r = y - one * mu
        Rinv_r = Rinv_y - Rinv_one * mu
        SigmaSqr = (r @ Rinv_r) / n

        if SigmaSqr <= 0 or not np.isfinite(SigmaSqr):
            return self.penalty, None, None

        negLnLike = (n / 2.0) * np.log(SigmaSqr) + 0.5 * LnDetR
        return negLnLike, None, None

    # -------- Unified likelihood --------

    def likelihood(self, x: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Computes the negative concentrated log-likelihood.
        Exact (Cholesky) if use_nystrom is False; otherwise Nyström.
        """
        if self.use_nystrom:
            return self._likelihood_nystrom(x)
        else:
            return self._likelihood_exact(x)

    # -------- Fit / optimize --------

    def max_likelihood(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """
        Hyperparameter estimation via maximum likelihood (Forrester, Ch. 3).
        """

        def objective(logtheta_loglambda_p_: np.ndarray) -> float:
            neg_ln_like, _, _ = self.likelihood(logtheta_loglambda_p_)
            return float(neg_ln_like)

        result = differential_evolution(func=objective, bounds=bounds, seed=self.seed)
        return result.x, result.fun

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None) -> "Kriging":
        """
        Fits the Kriging model. Public behavior matches the baseline class.
        Nyström path stores additional solver state for fast prediction.
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        self.X_ = X
        self.y_ = y
        self.n, self.k = self.X_.shape
        self._set_variable_types()
        if self.n_theta is None:
            self._set_theta()
        self.min_X = np.min(self.X_, axis=0)
        self.max_X = np.max(self.X_, axis=0)

        # Bounds: [θ] for interpolation; [θ, λ] for regression/reinterpolation; [+ p] if optim_p
        if bounds is None:
            if self.method == "interpolation":
                bounds = [(self.min_theta, self.max_theta)] * self.k
            else:
                bounds = [(self.min_theta, self.max_theta)] * self.k + [(self.min_Lambda, self.max_Lambda)]
        if self.optim_p:
            n_p = self.n_p if hasattr(self, "n_p") else self.k
            bounds += [(self.min_p, self.max_p)] * n_p

        # Optimize concentrated likelihood
        self.logtheta_loglambda_p_, _ = self.max_likelihood(bounds)

        # Store parameters on log10 scale; convert to linear only where needed
        self.theta = self.logtheta_loglambda_p_[: self.n_theta]
        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.Lambda = self.logtheta_loglambda_p_[self.n_theta : self.n_theta + 1]
            if self.optim_p:
                self.p_val = self.logtheta_loglambda_p_[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            self.Lambda = None
            if self.optim_p:
                self.p_val = self.logtheta_loglambda_p_[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        # Finalize model state at optimum:
        if self.use_nystrom:
            # Prepare Nyström state at the optimum
            if (self.method == "regression") or (self.method == "reinterpolation"):
                self.lambda_lin_ = float(10.0**self.Lambda)
            else:
                self.lambda_lin_ = float(self.eps)

            # Build W, C with final theta; store M chol
            self._nystrom_setup()
            C = self.C_
            W = self.W_chol_ @ self.W_chol_.T
            n = self.n
            m = W.shape[0]
            one = np.ones(n)
            yv = self.y_.flatten()

            M = W + (C.T @ C) / self.lambda_lin_
            try:
                self.M_chol_ = np.linalg.cholesky(M)
            except LinAlgError:
                self.M_chol_ = np.linalg.cholesky(M + 1e-12 * np.eye(m))

            # Precompute R^{-1}1, R^{-1}y, μ, σ^2, R^{-1}r
            Rinv_one = self._woodbury_solve(one)
            Rinv_y = self._woodbury_solve(yv)
            mu = (one @ Rinv_y) / (one @ Rinv_one)
            r = yv - one * mu
            Rinv_r = Rinv_y - Rinv_one * mu
            sigma2 = (r @ Rinv_r) / n

            # Store
            self.mu_ = float(mu)
            self.sigma2_ = float(max(0.0, sigma2))
            self.Rinv_one_ = Rinv_one
            self.Rinv_r_ = Rinv_r

            # Compute negLnLike for logging using Nyström
            negLnLike, _, _ = self._likelihood_nystrom(self.logtheta_loglambda_p_)
            self.negLnLike = float(negLnLike)
            self.Psi_, self.U_ = None, None
        else:
            # Exact final Psi and U
            self.negLnLike, self.Psi_, self.U_ = self._likelihood_exact(self.logtheta_loglambda_p_)

        self._update_log()
        return self

    # -------- Prediction --------

    def _pred_exact(self, x: np.ndarray) -> Tuple[float, float, Optional[float]]:
        y = self.y_.flatten()

        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.theta = self.logtheta_loglambda_p_[: self.n_theta]
            lambda_ = 10.0 ** self.logtheta_loglambda_p_[self.n_theta : self.n_theta + 1]
            if self.optim_p:
                self.p_val = self.logtheta_loglambda_p_[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            self.theta = self.logtheta_loglambda_p_[: self.n_theta]
            lambda_ = self.eps
            if self.optim_p:
                self.p_val = self.logtheta_loglambda_p_[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        U = self.U_
        n = self.X_.shape[0]
        one = np.ones(n)

        y_tilde = np.linalg.solve(U, y)
        y_tilde = np.linalg.solve(U.T, y_tilde)
        one_tilde = np.linalg.solve(U, one)
        one_tilde = np.linalg.solve(U.T, one_tilde)
        mu = (one @ y_tilde) / (one @ one_tilde)

        resid = y - one * mu
        resid_tilde = np.linalg.solve(U, resid)
        resid_tilde = np.linalg.solve(U.T, resid_tilde)

        psi = self.build_psi_vec(x)

        if (self.method == "interpolation") or (self.method == "regression"):
            SigmaSqr = (resid @ resid_tilde) / n
            psi_tilde = np.linalg.solve(U, psi)
            psi_tilde = np.linalg.solve(U.T, psi_tilde)
            SSqr = SigmaSqr * (1 + lambda_ - psi @ psi_tilde)
        else:
            Psi_adjusted = self.Psi_ - np.eye(n) * lambda_ + np.eye(n) * self.eps
            SigmaSqr = (resid @ np.linalg.solve(U.T, np.linalg.solve(U, Psi_adjusted @ resid_tilde))) / n
            Uint = np.linalg.cholesky(Psi_adjusted)
            psi_tilde = np.linalg.solve(Uint, psi)
            psi_tilde = np.linalg.solve(Uint.T, psi_tilde)
            SSqr = SigmaSqr * (1 - psi @ psi_tilde)

        s = float(np.abs(SSqr) ** 0.5)
        f = float(mu + psi @ resid_tilde)

        if self.return_ei:
            yBest = np.min(y)
            EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / (s + self.eps))))
            EITermTwo = (s + self.eps) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((yBest - f) ** 2 / ((s + self.eps) ** 2)))
            ExpImp = np.log10(EITermOne + EITermTwo + self.eps)
            return f, s, float(-ExpImp)
        else:
            return f, s, None

    def _pred_nystrom(self, x: np.ndarray) -> Tuple[float, float, Optional[float]]:
        """
        Prediction using Nyström state:
          f̂(x) = μ̂ + ψ(x)^T R^{-1} r
          s^2(x) = σ̂^2 [1 + λ − ψ(x)^T R^{-1} ψ(x)]
        """
        # Ensure state is available
        if self.C_ is None or self.M_chol_ is None or self.Rinv_r_ is None:
            # Fallback to exact if state missing
            return self._pred_exact(x)

        psi = self.build_psi_vec(x)
        # Apply R^{-1} to psi via Woodbury
        z = self._woodbury_solve(psi)
        f = float(self.mu_ + psi @ self.Rinv_r_)
        s2 = float(self.sigma2_ * (1.0 + self.lambda_lin_ - psi @ z))
        s = float(np.sqrt(max(0.0, s2)))

        if self.return_ei:
            y = self.y_.flatten()
            yBest = np.min(y)
            EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / (s + self.eps))))
            EITermTwo = (s + self.eps) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((yBest - f) ** 2 / ((s + self.eps) ** 2)))
            ExpImp = np.log10(EITermOne + EITermTwo + self.eps)
            return f, s, float(-ExpImp)
        else:
            return f, s, None

    def _pred(self, x: np.ndarray) -> Tuple[float, float, Optional[float]]:
        """
        Computes a single-point Kriging prediction (exact or Nyström).
        """
        if self.use_nystrom:
            return self._pred_nystrom(x)
        else:
            return self._pred_exact(x)

    def predict(self, X: np.ndarray, return_std=False, return_val: str = "y") -> np.ndarray:
        """
        Batch prediction wrapper around _pred, identical public behavior to baseline.
        """
        self.return_std = return_std
        X = self._reshape_X(X)

        if return_std:
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X])
            return np.array(predictions), np.array(std_devs)
        if return_val == "s":
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X])
            return np.array(std_devs)
        elif return_val == "all":
            self.return_std = True
            self.return_ei = True
            predictions, std_devs, eis = zip(*[self._pred(x_i) for x_i in X])
            return np.array(predictions), np.array(std_devs), np.array(eis)
        elif return_val == "ei":
            self.return_ei = True
            predictions, eis = zip(*[(self._pred(x_i)[0], self._pred(x_i)[2]) for x_i in X])
            return np.array(eis)
        else:
            predictions = [self._pred(x_i)[0] for x_i in X]
            return np.array(predictions)

    # -------- Plot (same behavior as baseline) --------

    def plot(self, i: int = 0, j: int = 1, show: Optional[bool] = True, add_points: bool = True) -> None:
        if self.k == 1:
            n_grid = 100
            x = linspace(self.min_X[0], self.max_X[0], num=n_grid).reshape(-1, 1)
            y = self.predict(x)
            plt.figure()
            plt.plot(x.ravel(), y, "k")
            if add_points:
                plt.plot(self.X_[:, 0], self.y_, "ro")
            plt.grid()
            if show:
                plt.show()
        else:
            plotkd(model=self, X=self.X_, y=self.y_, i=i, j=j, show=show, var_type=self.var_type, add_points=True)
