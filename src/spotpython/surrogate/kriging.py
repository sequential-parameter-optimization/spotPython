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


class Kriging(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible Kriging model class for regression tasks.
    Provides methods for likelihood evaluation, predictions, and hyperparameter optimization.

    Attributes:
        eps (float): A small regularization term to reduce ill-conditioning.
        penalty (float): The penalty value used if the correlation matrix is ill-conditioned.
        logtheta_lambda_ (np.ndarray): Best-fit log(theta) parameters from fit().
        U_ (np.ndarray): The Cholesky factor of the correlation matrix after fit().
        X_ (np.ndarray): The training input data (n x d).
        y_ (np.ndarray): The training target values (n,).
        negLnLike (float): The negative log-likelihood of the model.
        Psi_ (np.ndarray): The correlation matrix after fit().
        method (str): The fitting method used, can be "interpolation", "regression", or "reinterpolation".
        isotropic (bool): Whether the model is isotropic or not.

    Methods:
        __init__: Initializes the Kriging model with hyperparameters.
        _get_eps: Returns the square root of machine epsilon.
        _set_variable_types: Sets variable types for the model.
        get_model_params: Returns additional model parameters not included in get_params().
        _update_log: Updates the log with current model parameters.
        fit: Fits the Kriging model to training data X and y.
        predict: Predicts the Kriging response at a set of points X.
        build_Psi: Constructs a new correlation matrix Psi.
        likelihood: Computes the negative concentrated log-likelihood and correlation matrix.
        build_psi_vec: Builds the psi vector for predictive methods.
        _pred: Computes a single-point Kriging prediction.
    """

    def __init__(
        self,
        eps: float = None,
        penalty: float = 1e4,
        method="regression",
        noise: bool = False,
        var_type: List[str] = ["num"],
        name: str = "Kriging",
        seed: int = 124,
        model_optimizer=None,
        model_fun_evals: Optional[int] = None,
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
        metric_factorial="canberra",
        isotropic: bool = False,
        **kwargs,
    ):
        """
        Initializes the Kriging model.

        Args:
            eps (float, optional):
                Small number added to the diagonal of the correlation matrix to reduce
                ill-conditioning. Defaults to the square root of machine epsilon.
                Only used if method is "interpolation". Otherwise, if method is "regression" or "reinterpolation", eps is replaced by the
                lambda_ parameter. Defaults to None.
            penalty (float, optional):
                Large negative log-likelihood assigned if the correlation matrix is
                not positive-definite. Defaults to 1e4.
            method (str, optional):
                The type how the model uis fitted. Can be "interpolation", "regression", or "reinterpolation". Defaults to "regression".
            isotropic (bool, optional):
                If True, the model is isotropic, meaning all variables are treated equally (only one theta value is used).
                If False, the model can handle different theta values, one for each dimension. Defaults to False.
        """
        if eps is None:
            self.eps = self._get_eps()
        else:
            # check if eps is positive
            if eps <= 0:
                raise ValueError("eps must be positive")
            self.eps = eps
        self.penalty = penalty

        self.noise = noise
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
        self.n_theta = None  # Will be set in fit()
        self.isotropic = isotropic
        self.p_val = p_val
        self.n_p = n_p
        self.optim_p = optim_p
        self.theta_init_zero = theta_init_zero
        self.model_optimizer = model_optimizer
        if self.model_optimizer is None:
            self.model_optimizer = differential_evolution
        self.model_fun_evals = model_fun_evals
        if self.model_fun_evals is None:
            self.model_fun_evals = 100

        # Logging information
        self.log = {}
        self.log["negLnLike"] = []
        self.log["theta"] = []
        self.log["p"] = []
        self.log["Lambda"] = []

        self.logtheta_lambda_ = None
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

    def _get_eps(self) -> float:
        """
        Returns the square root of the machine epsilon.
        """
        eps = np.finfo(float).eps
        return np.sqrt(eps)

    def _set_variable_types(self) -> None:
        """
        Set the variable types for the class instance.
        This method sets the variable types for the class instance based
        on the `var_type` attribute. If the length of `var_type` is less
        than `k`, all variable types are forced to 'num' and a warning is logged.
        The method then creates Boolean masks for each variable
        type ('num', 'factor', 'int', 'ordered') using numpy arrays, e.g.,
        `num_mask = array([ True,  True])` if two numerical variables are present.

        Args:
            self (object): The Kriging object.

        Examples:
            >>> from spotpython.build import Kriging
                import numpy as np
                nat_X = np.array([[1, 2], [3, 4], [5, 6]])
                nat_y = np.array([1, 2, 3])
                var_type = ["num", "int", "float"]
                n_theta=2
                n_p=2
                S=Kriging(var_type=var_type, seed=124, n_theta=n_theta, n_p=n_p, optim_p=True, noise=True)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                assert S.var_type == ["num", "int", "float"]
                assert S.num_mask.all() == False
                assert S.factor_mask.all() == False
                assert S.int_mask.all() == False
                assert S.ordered_mask.all() == True
                assert np.all(S.num_mask == np.array([True, False, False]))
                assert np.all(S.int_mask == np.array([False, True, False]))
                assert np.all(S.ordered_mask == np.array([True, True, True]))

        Returns:
            None
        """
        # Ensure var_type has appropriate length by defaulting to 'num'
        if len(self.var_type) < self.k:
            self.var_type = ["num"] * self.k  # Corrected to fill with 'num' instead of duplicating
        # Create masks for each type using numpy vectorized operations
        var_type_array = np.array(self.var_type)
        self.num_mask = var_type_array == "num"
        self.factor_mask = var_type_array == "factor"
        self.int_mask = var_type_array == "int"
        self.ordered_mask = np.isin(var_type_array, ["int", "num", "float"])

    def get_model_params(self) -> Dict[str, float]:
        """
        Get the model parameters (in addition to sklearn's get_params method).

        This method is NOT required for scikit-learn compatibility.

        Returns:
            dict: Parameter names not included in get_params() mapped to their values.
        """
        return {"log_theta_lambda": self.logtheta_lambda_, "U": self.U_, "X": self.X_, "y": self.y_, "negLnLike": self.negLnLike}

    def _update_log(self) -> None:
        """
        If spot_writer is not None, this method writes the current values of
        negLnLike, theta, p (if optim_p is True),
        and Lambda (if method is not "interpolation") to the spot_writer object.

        Args:
            self (object): The Kriging object.

        Returns:
            None

        """
        self.log["negLnLike"] = append(self.log["negLnLike"], self.negLnLike)
        self.log["theta"] = append(self.log["theta"], self.theta)
        if self.optim_p:
            self.log["p"] = append(self.log["p"], self.p_val)
        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.log["Lambda"] = append(self.log["Lambda"], self.Lambda)
        # get the length of the log
        self.log_length = len(self.log["negLnLike"])
        if self.spot_writer is not None:
            negLnLike = self.negLnLike.copy()
            self.spot_writer.add_scalar("spot_negLnLike", negLnLike, self.counter + self.log_length)
            # add the self.n_theta theta values to the writer with one key "theta",
            # i.e, the same key for all theta values
            theta = self.theta.copy()
            self.spot_writer.add_scalars("spot_theta", {f"theta_{i}": theta[i] for i in range(self.n_theta)}, self.counter + self.log_length)
            if (self.method == "regression") or (self.method == "reinterpolation"):
                Lambda = self.Lambda.copy()
                self.spot_writer.add_scalar("spot_Lambda", Lambda, self.counter + self.log_length)
            if self.optim_p:
                p = self.p_val.copy()
                self.spot_writer.add_scalars("spot_p", {f"p_{i}": p[i] for i in range(self.n_p)}, self.counter + self.log_length)
            self.spot_writer.flush()

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None) -> "Kriging":
        """
        Fits the Kriging model to training data X and y. This method is compatible
        with scikit-learn and uses differential evolution to optimize the hyperparameters
        (log(theta)).

        Args:
            X (np.ndarray):
                Training input data of shape (n_samples, n_features).
            y (np.ndarray):
                Target values of shape (n_samples,) or (n_samples, 1).
            bounds (Optional[List[Tuple[float, float]]]):
                Bounds for each dimension of log(theta). If None, defaults to [(-3, 2)] * n_features.

        Returns:
            Kriging:
                The fitted Kriging model instance (self).

        Examples:
            >>> import numpy as np
            >>> from spotpython.surrogate.kriging import Kriging
            >>> # Training data
            >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> y_train = np.array([0.1, 0.2, 0.3])
            >>> # Initialize and fit the Kriging model
            >>> model = Kriging()
            >>> model.fit(X_train, y_train)
            >>> print("Fitted log(theta):", model.logtheta_lambda_)
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        self.X_ = X
        self.y_ = y
        self.n, self.k = self.X_.shape
        self._set_variable_types()
        if self.isotropic:
            # If isotropic, set n_theta to 1
            self.n_theta = 1
            print(f"Isotropic model: n_theta set to {self.n_theta}")
        else:
            self.n_theta = self.k
            print(f"Anisotropic model: n_theta set to {self.n_theta}")
        # Calculate and store min and max of X
        self.min_X = np.min(self.X_, axis=0)
        self.max_X = np.max(self.X_, axis=0)
        if bounds is None:
            if self.method == "interpolation":
                bounds = [(self.min_theta, self.max_theta)] * self.k
            else:
                # regression and reinterpolation use lambda_ as well
                bounds = [(self.min_theta, self.max_theta)] * self.k + [(self.min_Lambda, self.max_Lambda)]
        # Add p bounds if optimization is enabled
        if self.optim_p:
            # Number of p values to optimize (either 1 or k)
            n_p = self.n_p if hasattr(self, "n_p") else self.k
            bounds += [(self.min_p, self.max_p)] * n_p

        self.logtheta_lambda_, _ = self.max_likelihood(bounds)

        # store theta and Lambda in log scale
        if (self.method == "regression") or (self.method == "reinterpolation"):
            # select the first n_theta values from logtheta_lambda_:
            self.theta = self.logtheta_lambda_[: self.n_theta]
            self.Lambda = self.logtheta_lambda_[self.n_theta : self.n_theta + 1]
            if self.optim_p:
                self.p_val = self.logtheta_lambda_[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        else:
            self.theta = self.logtheta_lambda_[: self.n_theta]
            self.Lambda = None
            if self.optim_p:
                self.p_val = self.logtheta_lambda_[self.n_theta : self.n_theta + self.n_p]

        # Once logtheta_lambda is found, compute the final correlation matrix
        self.negLnLike, self.Psi_, self.U_ = self.likelihood(self.logtheta_lambda_)

        # Update log with the current values
        self._update_log()
        return self

    def predict(self, X: np.ndarray, return_std=False, return_val: str = "y") -> np.ndarray:
        """
        Predicts the Kriging response at a set of points X. This method is compatible
        with scikit-learn and returns predictions for the input points.

        Args:
            X (np.ndarray):
                Array of shape (n_samples, n_features) containing the points at which
                to predict the Kriging response.
            return_std (bool, optional):
                If True, returns the standard deviation of the predictions as well.
                Implememented for compatibility with scikit-learn.
                Defaults to False.
            return_val (str):
                Specifies which prediction values to return.
                It can be "y", "s", "ei", or "all".

        Returns:
            np.ndarray:
                Predicted values of shape (n_samples,).
            np.ndarray:
                If self.return_std is True, returns the standard deviations of the predictions
                of shape (n_samples,).

        Examples:
            >>> import numpy as np
            >>> from spotpython.surrogate.kriging import Kriging
            >>> # Training data
            >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            >>> y_train = np.array([0.1, 0.2, 0.3])
            >>> # Fit the Kriging model
            >>> model = Kriging().fit(X_train, y_train)
            >>> # Test data
            >>> X_test = np.array([[0.25, 0.25], [0.75, 0.75]])
            >>> # Predict responses
            >>> y_pred, sd, ei = model.predict(X_test)
            >>> print("Predictions:", y_pred)
        """
        self.return_std = return_std
        X = np.atleast_2d(X)
        if return_std:
            # Return predictions and standard deviations
            # Compatibility with scikit-learn
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X])
            return np.array(predictions), np.array(std_devs)
        if return_val == "s":
            # Return only standard deviations
            self.return_std = True
            predictions, std_devs = zip(*[self._pred(x_i)[:2] for x_i in X])
            return np.array(std_devs)
        elif return_val == "all":
            # Return predictions, standard deviations, and expected improvements
            self.return_std = True
            self.return_ei = True
            predictions, std_devs, eis = zip(*[self._pred(x_i) for x_i in X])
            return np.array(predictions), np.array(std_devs), np.array(eis)
        elif return_val == "ei":
            # Return only neg. expected improvements
            self.return_ei = True
            predictions, eis = zip(*[(self._pred(x_i)[0], self._pred(x_i)[2]) for x_i in X])
            return np.array(eis)
        else:
            # Return only predictions (case "y")
            predictions = [self._pred(x_i)[0] for x_i in X]
            return np.array(predictions)

    def build_Psi(self) -> None:
        """
        Constructs a new (n x n) correlation matrix Psi to reflect new data
        or a change in hyperparameters.
        This method uses `theta`, `p`, and coded `X` values to construct the
        correlation matrix as described in [Forr08a, p.57].

        Attributes:
            Psi (np.matrix): Correlation matrix Psi. Shape (n,n).
            cnd_Psi (float): Condition number of Psi.
            inf_Psi (bool): True if Psi is infinite, False otherwise.

        Raises:
            LinAlgError: If building Psi fails.

        Examples:
            >>> from spotpython.build.kriging import Kriging
                import numpy as np
                nat_X = np.array([[0], [1]])
                nat_y = np.array([0, 1])
                n=1
                p=1
                S=Kriging(name='kriging', seed=124, n_theta=n, n_p=p, optim_p=True, noise=False)
                S._initialize_variables(nat_X, nat_y)
                S._set_variable_types()
                print(S.nat_X)
                print(S.nat_y)
                S._set_theta_values()
                print(f"S.theta: {S.theta}")
                S._initialize_matrices()
                S._set_de_bounds()
                new_theta_p_Lambda = S._optimize_model()
                S._extract_from_bounds(new_theta_p_Lambda)
                print(f"S.theta: {S.theta}")
                S.build_Psi()
                print(f"S.Psi: {S.Psi}")
                    [[0]
                    [1]]
                    [0 1]
                    S.theta: [0.]
                    S.theta: [1.60036366]
                    S.Psi: [[1.00000001e+00 4.96525625e-18]
                    [4.96525625e-18 1.00000001e+00]]
        """
        try:
            n, k = self.X_.shape
            theta10 = np.power(10.0, self.theta)

            # Ensure theta has the correct length
            if self.n_theta == 1:
                theta10 = theta10 * np.ones(k)

            # Initialize the Psi matrix
            Psi = np.zeros((n, n), dtype=np.float64)

            # Calculate the distance matrix using ordered variables
            if self.ordered_mask.any():
                X_ordered = self.X_[:, self.ordered_mask]
                D_ordered = squareform(pdist(X_ordered, metric="sqeuclidean", w=theta10[self.ordered_mask]))
                Psi += D_ordered

            # Add the contribution of factor variables to the distance matrix
            if self.factor_mask.any():
                X_factor = self.X_[:, self.factor_mask]
                D_factor = squareform(pdist(X_factor, metric=self.metric_factorial, w=theta10[self.factor_mask]))
                Psi += D_factor

            # Calculate correlation from distance
            Psi = np.exp(-Psi)
            # Check for infinite values
            self.inf_Psi = np.isinf(Psi).any()
            # Calculate condition number
            self.cnd_Psi = cond(Psi)

            # Return only the upper triangle, as the matrix is symmetric
            # and the diagonal will be handled later.
            return np.triu(Psi, k=1)
        except LinAlgError as err:
            print("Building Psi failed. Error: %s, Type: %s", err, type(err))
            raise

    def likelihood(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes the negative of the concentrated log-likelihood for a given set
        of log(theta) parameters using a power exponent p=1.99. Returns the
        negative log-likelihood, the correlation matrix Psi, and its Cholesky factor U.

        Args:
            x (np.ndarray):
                1D array of log(theta) parameters of length k. If self.method is "regression" or
                "reinterpolation", length is k+1 and the last element of x is the log(noise) parameter.

        Returns:
            (float, np.ndarray, np.ndarray):
                (negLnLike, Psi, U) where:
                - negLnLike (float): The negative concentrated log-likelihood.
                - Psi (np.ndarray): The correlation matrix.
                - U (np.ndarray): The Cholesky factor (or None if ill-conditioned).
        """
        # Extract data
        X = self.X_
        y = self.y_.flatten()

        if (self.method == "regression") or (self.method == "reinterpolation"):
            # theta = x[:-1]
            self.theta = x[: self.n_theta]
            # lambda_ = x[-1]
            lambda_ = x[self.n_theta : self.n_theta + 1]
            # lambda is in log scale, so transform it back:
            lambda_ = 10.0**lambda_
            if self.optim_p:
                self.p_val = x[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            # theta = x
            self.theta = x[: self.n_theta]
            # use the original, untransformed eps:
            lambda_ = self.eps
            if self.optim_p:
                self.p_val = x[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        n = X.shape[0]
        one = np.ones(n)

        Psi_upper_triangle = self.build_Psi()

        Psi = Psi_upper_triangle + Psi_upper_triangle.T + np.eye(n) + np.eye(n) * lambda_

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

    def build_psi_vec(self, x: np.ndarray) -> None:
        """
        Build the psi vector required for predictive methods.

        Args:
            x (ndarray): Point to calculate the psi vector for.

        Returns:
            None

        Modifies:
            self.psi (np.ndarray): Updates the psi vector.

        Examples:
            >>> import numpy as np
                from spotpython.build.kriging import Kriging
                X_train = np.array([[1., 2.],
                                    [2., 4.],
                                    [3., 6.]])
                y_train = np.array([1., 2., 3.])
                S = Kriging(name='kriging',
                            seed=123,
                            log_level=50,
                            n_theta=1,
                            noise=False,
                            cod_type="norm")
                S.fit(X_train, y_train)
                # force theta to simple values:
                S.theta = np.array([0.0])
                nat_X = np.array([1., 0.])
                S.psi = np.zeros((S.n, 1))
                S.build_psi_vec(nat_X)
                res = np.array([[np.exp(-4)],
                    [np.exp(-17)],
                    [np.exp(-40)]])
                assert np.array_equal(S.psi, res)
                print(f"S.psi: {S.psi}")
                print(f"Control value res: {res}")
        """
        try:
            n = self.X_.shape[0]
            psi = np.zeros(n)
            theta10 = np.power(10.0, self.theta)
            if self.n_theta == 1:
                theta10 = theta10 * np.ones(self.k)

            D = np.zeros(n)

            # Compute ordered distance contributions
            if self.ordered_mask.any():
                X_ordered = self.X_[:, self.ordered_mask]
                x_ordered = x[self.ordered_mask]
                D += cdist(x_ordered.reshape(1, -1), X_ordered, metric="sqeuclidean", w=theta10[self.ordered_mask]).ravel()
            # Compute factor distance contributions
            if self.factor_mask.any():
                X_factor = self.X_[:, self.factor_mask]
                x_factor = x[self.factor_mask]
                D += cdist(x_factor.reshape(1, -1), X_factor, metric=self.metric_factorial, w=theta10[self.factor_mask]).ravel()

            psi = np.exp(-D)
            return psi

        except np.linalg.LinAlgError as err:
            print("Building psi failed due to a linear algebra error: %s. Error type: %s", err, type(err))

    def _pred(self, x: np.ndarray) -> float:
        """
        Computes a single-point Kriging prediction using the correlation matrix
        information. Internal helper method.

        Args:
            x (np.ndarray): 1D array of length k for the point at which to predict.

        Returns:
            float: The Kriging prediction at x.
            float: The standard deviation of the prediction.
            float: The NEGATIVE expected improvement at x.
        """
        y = self.y_.flatten()

        if (self.method == "regression") or (self.method == "reinterpolation"):
            self.theta = self.logtheta_lambda_[: self.n_theta]
            lambda_ = self.logtheta_lambda_[self.n_theta : self.n_theta + 1]
            # lambda is in log scale, so transform it back:
            lambda_ = 10.0**lambda_
            if self.optim_p:
                self.p_val = self.logtheta_lambda_[self.n_theta + 1 : self.n_theta + 1 + self.n_p]
        elif self.method == "interpolation":
            self.theta = self.logtheta_lambda_[: self.n_theta]
            # use the original, untransformed eps:
            lambda_ = self.eps
            if self.optim_p:
                self.p_val = self.logtheta_lambda_[self.n_theta : self.n_theta + self.n_p]
        else:
            raise ValueError("method must be one of 'interpolation', 'regression', or 'reinterpolation'")

        U = self.U_
        n = self.X_.shape[0]
        one = np.ones(n)

        # Compute mu
        y_tilde = np.linalg.solve(U, y)
        y_tilde = np.linalg.solve(U.T, y_tilde)
        one_tilde = np.linalg.solve(U, one)
        one_tilde = np.linalg.solve(U.T, one_tilde)
        mu = (one @ y_tilde) / (one @ one_tilde)

        resid = y - one * mu
        resid_tilde = np.linalg.solve(U, resid)
        resid_tilde = np.linalg.solve(U.T, resid_tilde)

        psi = self.build_psi_vec(x)

        # Compute SigmaSqr and SSqr
        if (self.method == "interpolation") or (self.method == "regression"):
            SigmaSqr = (resid @ resid_tilde) / n
            # Compute SSqr
            psi_tilde = np.linalg.solve(U, psi)
            psi_tilde = np.linalg.solve(U.T, psi_tilde)
            # Eq. (3.1) in [forr08a] without lambda:
            SSqr = SigmaSqr * (1 + lambda_ - psi @ psi_tilde)
        else:
            # method is "reinterpolation"
            Psi_adjusted = self.Psi_ - np.eye(n) * lambda_ + np.eye(n) * self.eps
            SigmaSqr = (resid @ np.linalg.solve(U.T, np.linalg.solve(U, Psi_adjusted @ resid_tilde))) / n
            # Compute Uint (Cholesky factor of the adjusted Psi matrix)
            Uint = np.linalg.cholesky(Psi_adjusted)

            # Compute SSqr
            psi_tilde = np.linalg.solve(Uint, psi)
            psi_tilde = np.linalg.solve(Uint.T, psi_tilde)
            SSqr = SigmaSqr * (1 - psi @ psi_tilde)

        # Compute s
        s = np.abs(SSqr) ** 0.5

        # Final prediction
        f = mu + psi @ resid_tilde

        # Compute ExpImp
        if self.return_ei:
            yBest = np.min(y)
            EITermOne = (yBest - f) * (0.5 + 0.5 * erf((1 / np.sqrt(2)) * ((yBest - f) / s)))
            EITermTwo = s * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((yBest - f) ** 2 / SSqr))
            ExpImp = np.log10(EITermOne + EITermTwo + self.eps)
            return float(f), float(s), float(-ExpImp)
        else:
            return float(f), float(s)

    def max_likelihood(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """
        Maximizes the Kriging likelihood function using differential evolution
        over the range of log(theta) specified by bounds.

        Args:
            bounds (List[Tuple[float, float]]): Sequence of (low, high) bounds for log(theta).

        Returns:
            (np.ndarray, float): (best_x, best_fun) where best_x is the
            optimal log(theta) array and best_fun is the minimized negative log-likelihood.
        """

        def objective(logtheta_lambda):
            neg_ln_like, _, _ = self.likelihood(logtheta_lambda)
            return neg_ln_like

        result = differential_evolution(objective, bounds)
        return result.x, result.fun

    def plot(self, i: int = 0, j: int = 1, show: Optional[bool] = True, add_points: bool = True) -> None:
        """
        This function plots 1D and 2D surrogates.
        Only for compatibility with the old Kriging implementation.

        Args:
            self (object):
                The Kriging object.
            i (int):
                The index of the first variable to plot.
            j (int):
                The index of the second variable to plot.
            show (bool):
                If `True`, the plots are displayed.
                If `False`, `plt.show()` should be called outside this function.
            add_points (bool):
                If `True`, the points from the design are added to the plot.

        Returns:
            None

        Note:
            * This method provides only a basic plot. For more advanced plots,
                use the `plot_contour()` method of the `Spot` class.

        Examples:
            >>> import numpy as np
                from spotpython.fun.objectivefunctions import Analytical
                from spotpython.spot import spot
                from spotpython.utils.init import fun_control_init, design_control_init
                # 1-dimensional example
                fun = analytical().fun_sphere
                fun_control=fun_control_init(lower = np.array([-1]),
                                            upper = np.array([1]),
                                            noise=False)
                design_control=design_control_init(init_size=10)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control)
                S.initialize_design()
                S.update_stats()
                S.fit_surrogate()
                S.surrogate.plot()
                # 2-dimensional example
                fun = analytical().fun_sphere
                fun_control=fun_control_init(lower = np.array([-1, -1]),
                                            upper = np.array([1, 1]),
                                            noise=False)
                design_control=design_control_init(init_size=10)
                S = spot.Spot(fun=fun,
                            fun_control=fun_control,
                            design_control=design_control)
                S.initialize_design()
                S.update_stats()
                S.fit_surrogate()
                S.surrogate.plot()
        """
        if self.k == 1:
            n_grid = 100
            x = linspace(self.min_X[0], self.max_X[0], num=n_grid)
            y = self.predict(x)
            plt.figure()
            plt.plot(x, y, "k")
            if show:
                plt.show()
        else:
            plotkd(model=self, X=self.X_, y=self.y_, i=i, j=j, show=show, var_type=self.var_type, add_points=True)
