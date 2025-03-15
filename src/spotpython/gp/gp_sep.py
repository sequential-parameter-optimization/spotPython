import math
import numpy as np
from spotpython.gp.covar import covar_sep_symm, covar_sep
from spotpython.gp.matrix import new_vector
from spotpython.gp.lite import predGPsep_lite
from spotpython.gp.likelihood import nlsep, gradnlsep
import warnings
import copy

# from scipy.spatial.distance import pdist, squareform
from spotpython.gp.distances import dist
from spotpython.utils.optimize import run_minimize_with_restarts
from spotpython.gp.distances import covar_anisotropic
from spotpython.utils.linalg import matrix_inversion_dispatcher
from numpy.linalg import det
from spotpython.utils.aggregate import select_distant_points
from sklearn.base import BaseEstimator, RegressorMixin


def crude_reset(theta, tmin, tmax, m) -> dict:
    """
    Check whether any elements of the parameter vector ``theta`` lie below the
    corresponding elements of the lower bound ``tmin``. If so, reset ``theta``
    to a new vector based on the weighted average of ``tmin`` and ``tmax``,
    leaving bounds unmodified except for cases where ``tmax`` is negative.

    Args:
        theta (np.ndarray): The current parameter values.
        tmin (np.ndarray): The lower bounds for the parameters.
        tmax (np.ndarray): The upper bounds for the parameters (may be adjusted if negative).
        m (int): The dimensionality or number of parameters (used to adjust negative ``tmax`` entries).

    Returns:
        (dict) or None: A dictionary containing:
            - "theta" (np.ndarray): The reset parameter values.
            - "its" (int): Number of iterations (0, indicating immediate reset).
            - "msg" (str): Reason for the reset.
            - "conv" (int): Reset code (102).
            Returns None if no reset is needed.
    """
    if np.any(theta < tmin):
        print("resetting due to init on lower boundary")
        print(f"theta: {theta}")
        print(f"tmin: {tmin}")
        for i in range(len(tmax)):
            if tmax[i] < 0:
                tmax[i] = np.sqrt(m)
        theta_new = 0.9 * np.maximum(tmin, 0) + 0.1 * np.array(tmax)
        return {
            "theta": theta_new,
            "its": 0,
            "msg": "reset due to init on lower boundary",
            "conv": 102,
        }
    return None


def getDs(X: np.ndarray, p: float = 0.1, samp_size: int = 1000) -> dict:
    """
    Calculate a rough starting, minimum, and maximum length-scale from the data X.

    Args:
        X (np.ndarray): The input data
        p (float): quantile for the distance distribution (default 0.1).
        samp_size (int): sub-sample size if the number of rows in X is large.

    Returns:
        dict: with 'start' (the p-th quantile),
                'min' (the minimum distance),
                'max' (the maximum distance).

    Examples:
        >>> from spotpython.gp.gp_sep import getDs
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> getDs(X, p=0.1, samp_size=10)
        >>> print(result)
    """
    if X is None:
        raise ValueError("The GP model does not have valid data to calculate distances.")

    # Sample rows if needed
    n = X.shape[0]
    X_sub = X
    if n > samp_size:
        idx = np.random.choice(n, samp_size, replace=False)
        X_sub = X_sub[idx, :]

    # Compute pairwise distances, get upper triangle, remove zeros
    # dist_matrix = squareform(pdist(X_sub))
    dist_matrix = dist(X_sub)
    iu = np.triu_indices(dist_matrix.shape[0], k=1)
    dvals = dist_matrix[iu]
    dvals = dvals[dvals > 0]

    # Calculate start, min, max
    dstart = np.quantile(dvals, p)
    dmin = np.min(dvals)
    dmax = np.max(dvals)

    return {"start": dstart, "min": dmin, "max": dmax}


def darg(d, X: np.ndarray = None, samp_size: int = 1000) -> dict:
    """
    Processes the 'd' dictionary/argument specifying length-scale priors,
    constraints, and whether MLE calculations should be used.

    Args:
        d (Union[Dict, float]): Could be a dictionary, numeric, or None.
        X (np.ndarray): The input data matrix.
        samp_size (int): The sub-sample size if the number of rows in X is large.

    Returns:
        dict: Updated 'd' with fields 'start', 'min', 'max', 'mle', 'ab', etc.

    Examples:
        >>> from spotpython.gp.gp_sep import darg
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> d = 2.5
        >>> result = darg(d=d, X=X, samp_size=10)
        >>> print(result)
    """
    if X is None:
        raise ValueError("The GP model does not have valid data to calculate distances.")

    # Coerce 'd' into a dict if necessary
    if d is None:
        d = {}
    elif isinstance(d, (int, float, np.number)):
        d = {"start": float(d)}
    elif not isinstance(d, dict):
        raise ValueError("d should be a dictionary, numeric, or None.")

    # Check for 'mle'
    if "mle" not in d:
        d["mle"] = True

    # Possibly build Ds from getDs if needed
    needsDs = ("start" not in d) or (d["mle"] and (("max" not in d) or ("min" not in d) or ("ab" not in d) or (d.get("ab", [None, None])[1] is None)))
    if needsDs:
        Ds = getDs(X=X, p=0.1, samp_size=samp_size)

    # Check for starting value
    if "start" not in d:
        d["start"] = Ds["start"]

    # Check for max value
    if "max" not in d:
        if d["mle"]:
            d["max"] = Ds["max"]
        else:
            d["max"] = float(np.max(d["start"]))

    # Check for min value
    if "min" not in d:
        if d["mle"]:
            d["min"] = Ds["min"] / 2.0
        else:
            d["min"] = float(np.min(d["start"]))
        if d["min"] < math.sqrt(np.finfo(float).eps):
            d["min"] = math.sqrt(np.finfo(float).eps)

    # Handle priors
    if not d["mle"]:
        d["ab"] = [0.0, 0.0]
    else:
        if "ab" not in d:
            d["ab"] = [1.5, None]
        if d["ab"][1] is None:
            # Placeholder logic
            d["ab"][1] = 0.5 / Ds["max"]

    # Basic range checks
    if d["max"] <= 0:
        raise ValueError("d['max'] should be > 0.")
    if d["min"] <= 0 or d["min"] > d["max"]:
        raise ValueError("d['min'] should be > 0 and < d['max'].")

    # Clamp 'start' into [min, max] rather than failing
    start_array = np.atleast_1d(d["start"])
    if np.any(start_array < d["min"]) or np.any(start_array > d["max"]):
        warnings.warn(f"Some 'start' values are out of [{d['min']}, {d['max']}]; " "clamping them to the valid range.", UserWarning)
        start_array = np.clip(start_array, d["min"], d["max"])

    # If start_array is length 1, store it back as a scalar
    d["start"] = start_array.item() if start_array.size == 1 else start_array

    # Minimal check for 'ab' (placeholder)
    ab_array = np.atleast_1d(d["ab"])
    if len(ab_array) != 2 or np.any(ab_array < 0):
        raise ValueError("d['ab'] must be a length-2, nonnegative array.")

    return d


def garg(g, y: np.ndarray = None) -> dict:
    """
    Process the 'g' argument to set up proper starting values, ranges,
    and priors for the nugget parameter.

    Args:
        g (dict}: Could be a dictionary, numeric, or None. If numeric, turn it into {"start": g}.
        y (np.ndarray): The response vector.

    Returns:
        dict: Updated 'g' with fields 'start', 'min', 'max', 'mle', 'ab', etc.
    """
    if y is None or len(y) == 0:
        raise ValueError("No response data found (y is empty).")

    # Coerce 'g' into a dict if necessary
    if g is None:
        g = {}
    elif isinstance(g, (int, float, np.number)):
        g = {"start": float(g)}
    elif not isinstance(g, dict):
        raise ValueError("g should be a dictionary, numeric, or None.")

    # Check for 'mle'
    if "mle" not in g:
        g["mle"] = False
    if not isinstance(g["mle"], bool):
        raise ValueError("g['mle'] should be a scalar boolean.")

    # Check if we need r2s (squared residuals)
    need_r2s = ("start" not in g) or (g["mle"] and (("max" not in g) or ("ab" not in g) or (g.get("ab", [None, None])[1] is None)))
    if need_r2s:
        r2s = (y - np.mean(y)) ** 2

    # Check for starting value
    if "start" not in g:
        g["start"] = float(np.quantile(r2s, 0.025))

    # Check for max value
    if "max" not in g:
        if g["mle"]:
            g["max"] = float(np.max(r2s))
        else:
            g["max"] = float(np.max(g["start"]))

    # Check for min value
    if "min" not in g:
        g["min"] = float(np.sqrt(np.finfo(float).eps))

    # Check for priors
    if not g["mle"]:
        g["ab"] = [0.0, 0.0]
    else:
        if "ab" not in g:
            g["ab"] = [1.5, None]
        if g["ab"][1] is None:
            s2max = float(np.mean(r2s))
            # Placeholder for Igamma.inv implementation
            g["ab"][1] = 0.5 / s2max  # simplified approximation

    # Basic range checks
    if g["max"] <= 0:
        raise ValueError("g['max'] should be > 0.")
    if g["min"] < 0 or g["min"] > g["max"]:
        raise ValueError("g['min'] should be >= 0 and <= g['max'].")

    # Clamp 'start' to valid range if needed
    start_array = np.atleast_1d(g["start"])
    if np.any(start_array < g["min"]) or np.any(start_array > g["max"]):
        warnings.warn(f"Some 'start' values are out of [{g['min']}, {g['max']}]; " "clamping them to the valid range.", UserWarning)
        start_array = np.clip(start_array, g["min"], g["max"])

    # If start_array is length 1, store it back as a scalar
    g["start"] = start_array.item() if start_array.size == 1 else start_array

    # Check ab
    ab_array = np.atleast_1d(g["ab"])
    if len(ab_array) != 2 or np.any(ab_array < 0):
        raise ValueError("g['ab'] must be a length-2, nonnegative array.")

    return g


class GPsep(BaseEstimator, RegressorMixin):
    """A class to represent a Gaussian Process with separable covariance.

    Attributes:
        m: Number of input dimensions.
        n: Number of observations.
        X: Input data matrix.
        y: Output data vector.
        d: Length-scale parameters.
        g: Nugget parameter.
        K: Covariance matrix.
        Ki: Inverse of covariance matrix.
        Kiy: Product of Ki and y.
        phi: Scalar value from y^T Ki y calculation.
        dK: Boolean flag for calculating derivatives.
        DK: Matrix of derivatives.
        ldetK: Log determinant of K.
        nlsep_method: Method for likelihood computation.
        gradnlsep_method: Method for gradient computation.
        n_restarts_optimizer: Number of restarts for optimization.
        samp_size: Sample size for distance calculations.
        maxit: Maximum number of optimization iterations.
        verbosity: Verbosity level.
        auto_optimize: Whether to automatically optimize hyperparameters.
        max_points: Maximum number of points for model building.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        d=None,
        g=None,
        nlsep_method="inv",
        gradnlsep_method="inv",
        n_restarts_optimizer=9,
        samp_size=1000,
        maxit=100,
        verbosity=0,
        auto_optimize=True,
        max_points=None,
        seed=123,
    ) -> None:
        """
        Initialize the GP model with data and hyperparameters.

        Args:
            d (np.ndarray):
                Length-scale parameters.
            g (float):
                Nugget parameter.
            nlsep_method (str):
                Method to use for likelihood optimization. Possible values are "inv" and "chol". Default is "inv".
            gradnlsep_method (str):
                Method to use for likelihood gradient optimization. Possible values are "inv", "chol", and "direct". Default is "inv".
            n_restarts_optimizer (int):
                Number of restarts for the optimizer. Default is 9.
            samp_size (int):
                sub-sample size for getDs(), darg() if the number of rows in X is large.
            maxit (int):
                Maximum number of iterations for the optimizer. Default is 100.
            verbosity (int):
                Verbosity level for optimization output. Default is 0.
            auto_optimize (bool):
                Whether to automatically optimize hyperparameters using MLE. Default is True.
            max_points (int):
                Maximum number of points to use for the model building. Default is None, which means all points are used.
            seed (int):
                Random seed for reproducibility. Default is 123.
        """
        # Hyperparameters (do not store training data)
        self.d = d
        self.g = g
        self.nlsep_method = nlsep_method
        self.gradnlsep_method = gradnlsep_method
        self.n_restarts_optimizer = n_restarts_optimizer
        self.samp_size = samp_size
        self.maxit = maxit
        self.verbosity = verbosity
        self.auto_optimize = auto_optimize
        self.max_points = max_points
        self.seed = seed

        # Attributes set during fit
        self.m = None
        self.n = None
        self.X_ = None
        self.y_ = None
        self.dk = None  # derivative flag
        self.K = None
        self.Ki = None
        self.Kiy = None
        self.phi = None
        self.dK = None
        self.DK = None
        self.ldetK = None

        # Internal flag to check if fitted
        self._is_fitted = False

        # need to store the initial parameters for the fit method (sklearn compatibility)
        self.init_params = {
            "d": d,
            "g": g,
            "nlsep_method": nlsep_method,
            "gradnlsep_method": gradnlsep_method,
            "n_restarts_optimizer": n_restarts_optimizer,
            "samp_size": samp_size,
            "maxit": maxit,
            "verbosity": verbosity,
            "auto_optimize": auto_optimize,
            "max_points": max_points,
            "seed": seed,
        }

    # Add these two methods required by scikit-learn
    def get_params(self, deep=True) -> dict:
        """Get parameters for this estimator.

        This method is required for scikit-learn compatibility.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators. Defaults to True.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            "d": self.d,
            "g": self.g,
            "nlsep_method": self.nlsep_method,
            "gradnlsep_method": self.gradnlsep_method,
            "n_restarts_optimizer": self.n_restarts_optimizer,
            "samp_size": self.samp_size,
            "maxit": self.maxit,
            "verbosity": self.verbosity,
            "auto_optimize": self.auto_optimize,
            "max_points": self.max_points,
            "seed": self.seed,
        }

    def set_params(self, **parameters: dict) -> "GPsep":
        """Set the parameters of this estimator.

        This method is required for scikit-learn compatibility.

        Args:
            **parameters (dict): Estimator parameters as keyword arguments.

        Returns:
            self (GPsep): Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        # Update the stored parameters for potential re-initialization
        self.init_params.update(parameters)

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, d=None, g=None, dK: bool = True, auto_optimize: bool = None, verbosity=0) -> "GPsep":
        """Fit the GP model with training data and optionally auto-optimize hyperparameters.

        Args:
            X (np.ndarray):
                Array-like of shape (n_samples, n_features).
            y (np.ndarray):
                Array-like of shape (n_samples,).
            d (Optional[Union[np.ndarray, float]]):
                The length-scale parameters. If None, will be determined
                automatically. Defaults to None.
            g (Optional[float]):
                The nugget parameter. If None, will be determined automatically. Defaults to None.
            dK (bool):
                Flag to indicate whether to calculate derivatives.
                Defaults to True.
            auto_optimize (Optional[bool]):
                Whether to automatically optimize hyperparameters
                using MLE. If None, uses the default value from the object.
                Defaults to None.
            verbosity (int):
                Verbosity level for optimization output. Defaults to 0.

        Returns:
            GPsep: The fitted GPsep object.

        Raises:
            ValueError: If X has no rows or if X and y dimensions mismatch.

        Examples:
            >>> from spotpython.gp.gp_sep import GPsep
            >>> import numpy as np
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> y = np.array([1, 2, 3])
            >>> model = GPsep()
            >>> model.fit(X, y)
        """
        # if X or y are pandas dataframes or series, convert them to numpy arrays
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        y = y.reshape(-1, 1)
        if verbosity > 0:
            print(f"X shape: {X.shape}, y shape: {y.shape}")
        if self.max_points is not None:
            if X.shape[0] > self.max_points:
                X, y = select_distant_points(X, y, self.max_points)
                if verbosity > 0:
                    print(f"Selected {self.max_points} points for the model.")
        if auto_optimize is None:
            auto_optimize = self.auto_optimize
        n, m = X.shape
        if n == 0:
            raise ValueError("X must be a matrix with rows.")
        if len(y) != n:
            raise ValueError(f"X has {n} rows but y length is {len(y)}")

        self.m = m
        self.n = n
        self.X = X
        self.y = y
        self.dk = dK

        # Determine good hyperparameters if not explicitly provided
        if d is None or g is None or auto_optimize:
            # Process length-scale arguments
            d_args = darg(d, X, samp_size=self.samp_size)

            # Process nugget arguments
            # TODO: Check if mle is True is correct
            g_dict = {"mle": True} if g is None else g
            g_args = garg(g_dict, y)

            # Use the determined parameters if not provided
            d_val = d_args["start"] if d is None else d
            g_val = g_args["start"] if g is None else g

            # Set the parameters
            self.d = np.full(m, d_val) if isinstance(d_val, (int, float)) else d_val
            if len(self.d) != m:
                raise ValueError(f"Length of d ({len(self.d)}) does not match ncol(X) ({m})")
            self.g = g_val

            if auto_optimize:
                tmin = [d_args["min"], g_args["min"]]  # Min bounds for d and g
                tmax = [d_args["max"], g_args["max"]]  # Max bounds for d and g
                ab = d_args["ab"] + g_args["ab"]  # Prior parameters (concatenated)
                # Check arguments and set defaults
                if tmin is None:
                    tmin = [np.sqrt(np.finfo(float).eps)] * 2
                if tmax is None:
                    tmax = [-1, 1]
                if ab is None:
                    ab = [0.0, 0.0, 0.0, 0.0]

                m = self.get_m()
                # Expand tmin, tmax if necessary
                if len(tmax) == 2:
                    tmax = [tmax[0]] * m + [tmax[1]]
                elif len(tmax) != m + 1:
                    raise ValueError("length(tmax) must be 2 or m+1")

                if len(tmin) == 2:
                    tmin = [tmin[0]] * m + [tmin[1]]
                elif len(tmin) != m + 1:
                    raise ValueError("length(tmin) must be 2 or m+1")

                if len(ab) != 4 or any(val < 0 for val in ab):
                    raise ValueError("ab must be a list of four non-negative numbers")

                # Possibly reset parameters
                theta = np.concatenate((self.get_d(), [self.get_g()]))
                # Check if theta is on the boundary. If not on the boundary,
                # reset the  current parameters.
                theta_new = crude_reset(theta, tmin, tmax, m)
                if theta_new is not None:
                    theta = theta_new["theta"]
                    # isuue a warning if the parameters are reset
                    warnings.warn(f"resetting due to init on lower boundary: {theta_new['msg']}", RuntimeWarning)

                # Convert ab to numpy array if it is a list
                if not isinstance(ab, np.ndarray):
                    ab = np.array(ab, dtype=float)

                # check leghtscale bounds:
                for j in range(self.m):
                    if tmin[j] <= 0:
                        tmin[j] = np.finfo(float).eps
                    if tmax[j] <= 0:
                        tmax[j] = self.m**2
                    if self.d[j] > tmax[j]:
                        raise ValueError(f"d[{j}]={self.d[j]} > tmax[{j}]={tmax[j]}")
                    elif self.d[j] < tmin[j]:
                        raise ValueError(f"d[{j}]={self.d[j]} < tmin[{j}]={tmin[j]}")

                # check nugget bounds
                if tmin[self.m] <= 0:
                    tmin[self.m] = np.finfo(float).eps
                if self.g > tmax[self.m]:
                    raise ValueError(f"g={self.g} > tmax={tmax[self.m]}")
                elif self.g < tmin[self.m]:
                    raise ValueError(f"g={self.g} < tmin={tmin[self.m]}")

                # Check for negative entries in ab array
                if np.any(ab < 0):
                    raise ValueError("ab must be a positive 4-vector")

                # TODO: check if this is necessary
                # if self.DK is None:
                #     raise ValueError("derivative info not in GPsep; use newGPsep with dK=True")

                # New: mleGPsep_optimize starts here:

                # generate starting point p
                p = np.concatenate([self.d, [self.g]])
                bounds = [(tmin[i], tmax[i]) for i in range(len(p))]
                if self.verbosity > 0:
                    print(f"Starting MLE with d={self.d}, g={self.g}")
                    print(f"Starting point: {p}")
                    print(f"bounds: {bounds}")
                    print(f"p: {p}")
                X = copy.deepcopy(self.X)
                y = copy.deepcopy(self.y)

                def objective(par):
                    return nlsep(par, X, y, self.nlsep_method)

                def gradient(par):
                    return gradnlsep(par, X, y, self.gradnlsep_method)

                result = run_minimize_with_restarts(
                    objective=objective, gradient=gradient, x0=p, bounds=bounds, n_restarts_optimizer=self.n_restarts_optimizer, maxit=self.maxit, verb=self.verbosity, random_state=self.seed
                )

                d = result.x[:-1]
                g = result.x[-1]

                # set new parameters and build
                self.set_new_params(d, g)
                if self.verbosity > 0:
                    print(f"result: {result}")
                    print(f"Optimized d: {d}, g: {g}")
                    print(f"Updated d: {self.d}, g: {self.g}")
                self._build()
                new_theta = np.concatenate((self.get_d(), [self.get_g()]))
                if np.sqrt(np.mean((result.x - new_theta) ** 2)) > np.sqrt(np.finfo(float).eps):
                    warnings.warn("stored theta not the same as theta-hat", RuntimeWarning)
                if verbosity > 0:
                    # Print mle optimization results
                    print("MLE Optimization complete:")
                    print(f"Optimized lengthscale (d): {self.get_d()}")
                    print(f"Optimized nugget (g): {self.get_g()}")
                    print(f"Message: {result['msg']}")
                    print(f"Iterations: {result['its']}")
                self._is_fitted = True
                return self
            else:
                # No optimization, just build the model with roughly estimated parameters using darg and garg
                self._build()
                self._is_fitted = True
                return self
        else:
            # Original behavior for explicitly provided parameters
            print("Using provided hyperparameters.")
            self.d = np.full(m, d) if isinstance(d, (int, float)) else d
            if len(self.d) != m:
                raise ValueError(f"Length of d ({len(self.d)}) does not match ncol(X) ({m})")
            self.g = g
            self._build()
            self._is_fitted = True
            return self

    def calc_ytKiy(self) -> None:
        """
        Recalculate phi and related components from Ki and y.
        """
        if self.Kiy is None:
            self.Kiy = new_vector(self.n)

        # Convert y to numpy array if it's a pandas Series
        if hasattr(self.y, "to_numpy"):
            y_array = self.y.to_numpy()
        else:
            y_array = np.asarray(self.y)

        y = y_array.reshape(-1, 1)
        Kiy = np.dot(self.Ki, y)
        phi = np.dot(y.T, Kiy)
        self.phi = phi[0, 0]
        self.Kiy = Kiy

    def _build(self) -> None:
        """
        Completes all correlation calculations after data is defined.
        """
        # TODO: check if the following line is necessary
        # if self.K is not None:
        #     raise RuntimeError("Covariance matrix has already been built.")
        self.K = covar_anisotropic(self.X, d=self.d, g=self.g)
        self.Ki = matrix_inversion_dispatcher(self.K, method=self.nlsep_method)
        detK = det(self.K)
        if detK <= 1e-14:
            detK = 1e-14  # TODO: Check if this can be improved
        self.ldetK = np.log(detK)
        self.calc_ytKiy()
        # TODO: Check if this is necessary
        # if self.dK:
        #     # TODO: Check if this is necessary
        #     # if self.dK is not None:
        #     #     raise RuntimeError("dK calculations have already been initialized.")
        #     self.DK = diff_covar_sep_symm(self.m, self.X, self.n, self.d, self.K)

    def _check_is_fitted(self) -> None:
        """
        Check if the GPsep instance is fitted.
        """
        if not self._is_fitted:
            raise ValueError("This GPsep instance is not fitted yet. Call 'fit' with " "appropriate arguments before using 'predict'.")

    def predict(self, X: np.ndarray, lite: bool = False, nonug: bool = False, return_full=False, return_std=False) -> float:
        """Predict the Gaussian Process output at new input points.

        Args:
            X (np.ndarray):
                The predictive locations.
            lite (bool):
                Flag to indicate whether to compute only the diagonal
                of Sigma. Defaults to False.
            nonug (bool):
                Flag to indicate whether to exclude nugget.
                Defaults to False.
            return_full (bool):
                Flag to indicate whether to return the full dictionary,
                which includes the mean, Sigma, df, and llik. Defaults to False.
            return_std (bool):
                Flag to indicate whether to return the standard deviation.
                Only applicable when return_full is False. Defaults to False.

        Returns:
            Various formats based on arguments:
            - If return_full=True: Dictionary with 'mean', 'Sigma'/'s2', 'df', 'llik'
            - If return_std=True: Tuple (mean, std_deviation)
            - Otherwise: Mean predictions

        Examples:
            import numpy as np
            from spotpython.gp.gp_sep import newGPsep
            import matplotlib.pyplot as plt
            # Simple sine data
            X = np.linspace(0, 2 * np.pi, 7).reshape(-1, 1)
            y = np.sin(X)
            # New GP fit
            gpsep = newGPsep(X, y, d=2, g=0.000001)
            # Make predictions
            XX = np.linspace(-1, 2 * np.pi + 1, 499).reshape(-1, 1)
            p = gpsep.predict(XX, lite=False)
            # Sample from the predictive distribution
            N = 100
            mean = p["mean"]
            Sigma = p["Sigma"]
            df = p["df"]
            # Generate samples from the multivariate t-distribution
            yy = np.random.multivariate_normal(mean, Sigma, N)
            yy = yy.T
            # Plot the results
            plt.figure(figsize=(10, 6))
            for i in range(N):
                plt.plot(XX, yy[:, i], color="gray", linewidth=0.5)
            plt.scatter(X, y, color="black", s=50, zorder=5)
            plt.xlabel("x")
            plt.ylabel("f-hat(x)")
            plt.title("Predictive Distribution")
            plt.show()
        """
        self._check_is_fitted()
        # if X is a pandas dataframe, convert it to a numpy array
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if lite:
            res = self._predict_lite(X, nonug)
            if return_full:
                return res
            elif return_std:
                return (res["mean"], res["s2"])
            else:
                return res["mean"]
        else:
            res = self._predict_full(X, nonug)
            if return_full:
                return res
            elif return_std:
                return (res["mean"], res["Sigma"])
            else:
                return res["mean"]

    def _predict_lite(self, X: np.ndarray, nonug: bool) -> dict:
        """
        Predict only the diagonal of Sigma—optimized for speed.

        Args:
            X (np.ndarray): The predictive locations.
            nonug (bool): Flag to indicate whether to use nugget.

        Returns:
            dict: A dictionary containing the mean, s2, df, and llik.
        """
        nn = X.shape[0]
        m = X.shape[1]
        mean_out, s2_out, df_out, llik_out = predGPsep_lite(self, m, nn, X, lite_in=True, nonug_in=nonug)
        return {"mean": mean_out, "s2": s2_out, "df": df_out, "llik": llik_out}

    def _predict_full(self, X: np.ndarray, nonug: bool) -> dict:
        """
        Compute full predictive covariance matrix.

        Args:
            X (np.ndarray): The predictive locations.
            nonug (bool): Flag to indicate whether to use nugget.

        Returns:
            dict: A dictionary containing the mean, Sigma, df, and llik.
        """
        nn, m = X.shape
        if m != self.m:
            raise ValueError(f"ncol(X)={m} does not match GPsep model ({self.m})")

        mean = np.zeros(nn)
        Sigma = np.zeros((nn, nn))
        df = np.zeros(1)
        llik = np.zeros(1)

        n = self.n
        g = np.finfo(float).eps if nonug else self.g
        df[0] = float(n)
        phidf = self.phi / df[0]
        llik[0] = -0.5 * (df[0] * np.log(0.5 * self.phi) + self.ldetK)
        k = covar_sep(self.m, self.X, n, X, nn, self.d, 0.0)
        Sigma[...] = covar_sep_symm(self.m, X, nn, self.d, g)
        ktKi = np.dot(k.T, self.Ki)
        mean[:] = np.dot(ktKi, self.y).reshape(-1)
        Sigma[...] = phidf * (Sigma - np.dot(ktKi, k))
        return {"mean": mean, "Sigma": Sigma, "df": df, "llik": llik}

    def get_d(self) -> np.ndarray:
        """
        Access the separable lengthscale parameter of the GP.

        Returns:
            np.ndarray: The lengthscale parameter.
        """
        if self.d is None:
            raise ValueError("Lengthscale parameter d is not allocated.")
        return np.copy(self.d)

    def get_g(self) -> float:
        """
        Access the nugget parameter of the GP.

        Returns:
            float: The nugget parameter.
        """
        if self.g is None:
            raise ValueError("Nugget parameter g is not allocated.")
        return self.g

    def get_m(self) -> int:
        """
        Access the input dimension m of the GP.

        Returns:
            int: The input dimension m.
        """
        if self.m is None:
            raise ValueError("Input dimension m is not allocated.")
        return self.m

    def set_new_params(self, d: np.ndarray, g: float) -> None:
        """
        Change the parameterization of the GP without destroying and reallocating memory.

        Args:
            d (np.ndarray): The new length-scale parameters.
            g (float): The new nugget parameter.
        """
        if self.d is None or self.g is None:
            raise ValueError("GP parameters are not allocated.")

        dsame = np.allclose(self.d, d)
        if dsame and g == self.g:
            return

        self.d = np.where(d <= 0, self.d, d)
        self.g = g if g >= 0 else self.g

    def mleGPsep_optimize(self, tmin: np.ndarray, tmax: np.ndarray, ab: np.ndarray, maxit: int, verb: int) -> dict:
        """
        Optimize the separable GP to use its MLE separable lengthscale and multiple nugget parameterization using the current data.

        Args:
            tmin (np.ndarray): Minimum bounds for the parameters.
            tmax (np.ndarray): Maximum bounds for the parameters.
            ab (np.ndarray): Prior parameters. Currently unused.
            maxit (int): Maximum number of iterations.
            verb (int): Verbosity level.

        Returns:
            dict: A dictionary containing the optimized parameters, number of iterations, convergence status, and message.
        """
        print(f"Starting MLE with d={self.d}, g={self.g}")
        # generate starting point p
        p = np.concatenate([self.d, [self.g]])
        print(f"Starting point: {p}")
        bounds = [(tmin[i], tmax[i]) for i in range(len(p))]
        print(f"bounds: {bounds}")

        def objective(par):
            return nlsep(par, self.X, self.y, self.nlsep_method)

        def gradient(par):
            return gradnlsep(par, self.X, self.y, self.gradnlsep_method)

        result = run_minimize_with_restarts(objective=objective, gradient=gradient, x0=p, bounds=bounds, n_restarts_optimizer=self.n_restarts_optimizer, maxit=maxit, verb=verb)

        print(f"result: {result}")

        d = result.x[:-1]
        g = result.x[-1]
        print(f"Optimized d: {d}, g: {g}")
        # set new parameters and build
        self.set_new_params(d, g)
        print(f"Updated d: {self.d}, g: {self.g}")
        self._build()
        return {"parameters": result.x, "iterations": result.nit, "convergence": result.status, "message": result.message}


def newGPsep(X: np.ndarray, y: np.ndarray, d=None, g=None, dK: bool = True, optimize: bool = True) -> GPsep:
    """
    Instantiate a new GPsep model with automatic hyperparameter optimization.

    Args:
        X (np.ndarray): The input data matrix of shape (n, m).
        y (np.ndarray): The output data vector of length n.
        d (optional): The length-scale parameters. If None, will be determined automatically.
        g (optional): The nugget parameter. If None, will be determined automatically.
        dK (bool): Flag to indicate whether to calculate derivatives.
        optimize (bool): Whether to optimize hyperparameters after initialization.

    Returns:
        GPsep: The newly created and optimized GPsep object.
    """
    gpsep = GPsep()
    return gpsep.fit(X, y, d=d, g=g, dK=dK, auto_optimize=optimize)
