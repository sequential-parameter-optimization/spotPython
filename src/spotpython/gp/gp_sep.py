import math
import numpy as np
from spotpython.gp.covar import covar_sep_symm, covar_sep, diff_covar_sep_symm
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
        >>> from spotpython.gp.gp_sep import GPsep
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> gp = GPsep(m=2, n=3, X=X)
        >>> d = 2.5
        >>> result = gp.darg(d=d, X=X, samp_size=10)
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
        g: Could be a dictionary, numeric, or None. If numeric, turn it into {"start": g}.
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


class GPsep:
    """A class to represent a Gaussian Process with separable covariance."""

    def __init__(
        self,
        X: np.ndarray = None,
        Z: np.ndarray = None,
        d: np.ndarray = None,
        g: float = None,
        nlsep_method="inv",
        gradnlsep_method="inv",
        n_restarts_optimizer=9,
        samp_size: int = 1000,
        maxit=100,
        verbosity=0,
        auto_optimize=True,
        max_points=None,
    ) -> None:
        """
        Initialize the GP model with data and hyperparameters.

        Args:
            X (np.ndarray):
                Input data matrix of shape (n, m). If pandas DataFrame, will be converted to numpy array.
            Z (np.ndarray):
                Output data vector of length n. If pandas Series, will be converted to numpy array.
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
        """
        if X is not None:
            # convert pandas dataframes or series to numpy arrays
            if hasattr(X, "to_numpy"):
                X = X.to_numpy()
        if Z is not None:
            if hasattr(Z, "to_numpy"):
                Z = Z.to_numpy()
            Z = Z.reshape(-1, 1)
        if X is not None and Z is not None:
            if max_points is not None:
                X, Z = select_distant_points(X, Z, max_points)
                print(f"Selected {max_points} points for the model.")
        self.m = None  # (int) number of input dimensions
        self.n = None  # (int) number of observations
        self.X = X
        self.Z = Z
        self.d = d
        self.g = g
        self.K = None
        self.Ki = None
        self.KiZ = None
        self.phi = None
        self.dK = None  # boolean
        self.DK = None  # matrix
        self.ldetK = None
        self.nlsep_method = nlsep_method
        self.gradnlsep_method = gradnlsep_method
        self.n_restarts_optimizer = n_restarts_optimizer
        self.samp_size = samp_size
        self.maxit = maxit
        self.verbosity = verbosity
        self.auto_optimize = auto_optimize
        self.max_points = max_points

    def fit(self, X: np.ndarray, Z: np.ndarray, d=None, g=None, dK: bool = True, auto_optimize: bool = None, verbosity=0) -> "GPsep":
        """
        Fits the GP model with training data and optionally auto-optimizes hyperparameters.

        Args:
            X (np.ndarray): The input data matrix of shape (n, m).
            Z (np.ndarray): The output data vector of length n.
            d (Union[np.ndarray, float, None]): The length-scale parameters. If None, will be determined automatically.
            g (Union[float, None]): The nugget parameter. If None, will be determined automatically.
            dK (bool): Flag to indicate whether to calculate derivatives. Default is True.
            auto_optimize (bool): Whether to automatically optimize hyperparameters using MLE.
            verbosity (int): Verbosity level for optimization output.
            auto_optimize (bool): Whether to automatically optimize hyperparameters using MLE. If None, uses the default value from the object, which is True.

        Returns:
            GPsep: The fitted GPsep object.
        """
        # if X or Z are pandas dataframes or series, convert them to numpy arrays
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(Z, "to_numpy"):
            Z = Z.to_numpy()
        Z = Z.reshape(-1, 1)
        print(f"X shape: {X.shape}, Z shape: {Z.shape}")
        if self.max_points is not None:
            X, Z = select_distant_points(X, Z, self.max_points)
            print(f"Selected {self.max_points} points for the model.")
        if auto_optimize is None:
            auto_optimize = self.auto_optimize
        n, m = X.shape
        if n == 0:
            raise ValueError("X must be a matrix with rows.")
        if len(Z) != n:
            raise ValueError(f"X has {n} rows but Z length is {len(Z)}")

        self.m = m
        self.n = n
        self.X = X
        self.Z = Z
        self.dk = dK

        # Determine good hyperparameters if not explicitly provided
        if d is None or g is None or auto_optimize:
            # Process length-scale arguments
            d_args = darg(d, X, samp_size=self.samp_size)

            # Process nugget arguments
            # TODO: Check if mle is True is correct
            g_dict = {"mle": True} if g is None else g
            g_args = garg(g_dict, Z)

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
                # build the model and return the current parameters.
                if np.any(theta < tmin):
                    print("resetting due to init on lower boundary")
                    print(f"theta: {theta}")
                    print(f"tmin: {tmin}")
                    for i in range(len(tmax)):
                        if tmax[i] < 0:
                            tmax[i] = np.sqrt(m)
                    theta_new = 0.9 * np.maximum(tmin, 0) + 0.1 * np.array(tmax)
                    self.set_new_params(theta_new[:m], theta_new[m])
                    self.build()
                    return {
                        "theta": theta_new,
                        "its": 0,
                        "msg": "reset due to init on lower boundary",
                        "conv": 102,
                    }
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
                Z = copy.deepcopy(self.Z)

                def objective(par):
                    return nlsep(par, X, Z, self.nlsep_method)

                def gradient(par):
                    return gradnlsep(par, X, Z, self.gradnlsep_method)

                result = run_minimize_with_restarts(objective=objective, gradient=gradient, x0=p, bounds=bounds, n_restarts_optimizer=self.n_restarts_optimizer, maxit=self.maxit, verb=self.verbosity)

                d = result.x[:-1]
                g = result.x[-1]


                # set new parameters and build
                self.set_new_params(d, g)
                if self.verbosity > 0:
                    print(f"result: {result}")
                    print(f"Optimized d: {d}, g: {g}")
                    print(f"Updated d: {self.d}, g: {self.g}")
                self.build()
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
                return self
            else:
                # No optimization, just build the model with roughly estimated parameters using darg and garg
                self.build()
                return self
        else:
            # Original behavior for explicitly provided parameters
            print("Using provided hyperparameters.")
            self.d = np.full(m, d) if isinstance(d, (int, float)) else d
            if len(self.d) != m:
                raise ValueError(f"Length of d ({len(self.d)}) does not match ncol(X) ({m})")
            self.g = g
            self.build()
            return self

    def calc_ZtKiZ(self) -> None:
        """
        Recalculate phi and related components from Ki and Z.
        """
        if self.KiZ is None:
            self.KiZ = new_vector(self.n)

        # Convert Z to numpy array if it's a pandas Series
        if hasattr(self.Z, "to_numpy"):
            Z_array = self.Z.to_numpy()
        else:
            Z_array = np.asarray(self.Z)

        Z = Z_array.reshape(-1, 1)
        KiZ = np.dot(self.Ki, Z)
        phi = np.dot(Z.T, KiZ)
        self.phi = phi[0, 0]
        self.KiZ = KiZ

    def build(self) -> None:
        """
        Completes all correlation calculations after data is defined.
        """
        # TODO: check if the following line is necessary
        # if self.K is not None:
        #     raise RuntimeError("Covariance matrix has already been built.")
        self.K = covar_anisotropic(self.X, d=self.d, g=self.g)
        self.Ki = matrix_inversion_dispatcher(self.K, method=self.nlsep_method)
        self.ldetK = np.log(det(self.K))
        self.calc_ZtKiZ()
        if self.dK:
            # TODO: Check if this is necessary
            # if self.dK is not None:
            #     raise RuntimeError("dK calculations have already been initialized.")
            self.DK = diff_covar_sep_symm(self.m, self.X, self.n, self.d, self.K)

    def predict(self, XX: np.ndarray, lite: bool = False, nonug: bool = False, return_full=False, return_std=False) -> float:
        """
        Predict the Gaussian Process output at new input points.

        Args:
            XX (np.ndarray):
                The predictive locations.
            lite (bool):
                Flag to indicate whether to compute only the diagonal of Sigma.
            nonug (bool):
                Flag to indicate whether to use nugget.
            return_full (bool): Flag to indicate whether to return the full dictionry, which
                includes the mean, Sigma, df, and llik. Default is False.
            return_std (bool):
                Flag to indicate whether to return the standard deviation. Only applicable when
                return_full is False. Default is False.

        Returns:
            float:
                The predicted output at the new input points.
                If return_full is True, returns a containing the mean, Sigma (or s2), df, and llik.
                If return_std is True, returns a tuple containing the mean and standard deviation.

        Examples:
                import numpy as np
                from spotpython.gp.gp_sep import newGPsep
                import matplotlib.pyplot as plt
                # Simple sine data
                X = np.linspace(0, 2 * np.pi, 7).reshape(-1, 1)
                Z = np.sin(X)
                # New GP fit
                gpsep = newGPsep(X, Z, d=2, g=0.000001)
                # Make predictions
                XX = np.linspace(-1, 2 * np.pi + 1, 499).reshape(-1, 1)
                p = gpsep.predict(XX, lite=False)
                # Sample from the predictive distribution
                N = 100
                mean = p["mean"]
                Sigma = p["Sigma"]
                df = p["df"]
                # Generate samples from the multivariate t-distribution
                ZZ = np.random.multivariate_normal(mean, Sigma, N)
                ZZ = ZZ.T
                # Plot the results
                plt.figure(figsize=(10, 6))
                for i in range(N):
                    plt.plot(XX, ZZ[:, i], color="gray", linewidth=0.5)
                plt.scatter(X, Z, color="black", s=50, zorder=5)
                plt.xlabel("x")
                plt.ylabel("f-hat(x)")
                plt.title("Predictive Distribution")
                plt.show()
        """
        # if XX is a pandas dataframe, convert it to a numpy array
        if hasattr(XX, "to_numpy"):
            XX = XX.to_numpy()
        if lite:
            res = self._predict_lite(XX, nonug)
            if return_full:
                return res
            elif return_std:
                return (res["mean"], res["s2"])
            else:
                return res["mean"]
        else:
            res = self._predict_full(XX, nonug)
            if return_full:
                return res
            elif return_std:
                return (res["mean"], res["Sigma"])
            else:
                return res["mean"]

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

        mean = np.zeros(nn)
        Sigma = np.zeros((nn, nn))
        df = np.zeros(1)
        llik = np.zeros(1)

        n = self.n
        g = np.finfo(float).eps if nonug else self.g
        df[0] = float(n)
        phidf = self.phi / df[0]
        llik[0] = -0.5 * (df[0] * np.log(0.5 * self.phi) + self.ldetK)
        k = covar_sep(self.m, self.X, n, XX, nn, self.d, 0.0)
        Sigma[...] = covar_sep_symm(self.m, XX, nn, self.d, g)
        ktKi = np.dot(k.T, self.Ki)
        mean[:] = np.dot(ktKi, self.Z).reshape(-1)
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
            return nlsep(par, self.X, self.Z, self.nlsep_method)

        def gradient(par):
            return gradnlsep(par, self.X, self.Z, self.gradnlsep_method)

        result = run_minimize_with_restarts(objective=objective, gradient=gradient, x0=p, bounds=bounds, n_restarts_optimizer=self.n_restarts_optimizer, maxit=maxit, verb=verb)

        print(f"result: {result}")

        d = result.x[:-1]
        g = result.x[-1]
        print(f"Optimized d: {d}, g: {g}")
        # set new parameters and build
        self.set_new_params(d, g)
        print(f"Updated d: {self.d}, g: {self.g}")
        self.build()
        return {"parameters": result.x, "iterations": result.nit, "convergence": result.status, "message": result.message}


def newGPsep(X: np.ndarray, Z: np.ndarray, d=None, g=None, dK: bool = True, optimize: bool = True) -> GPsep:
    """
    Instantiate a new GPsep model with automatic hyperparameter optimization.

    Args:
        X (np.ndarray): The input data matrix of shape (n, m).
        Z (np.ndarray): The output data vector of length n.
        d (optional): The length-scale parameters. If None, will be determined automatically.
        g (optional): The nugget parameter. If None, will be determined automatically.
        dK (bool): Flag to indicate whether to calculate derivatives.
        optimize (bool): Whether to optimize hyperparameters after initialization.

    Returns:
        GPsep: The newly created and optimized GPsep object.
    """
    gpsep = GPsep()
    return gpsep.fit(X, Z, d=d, g=g, dK=dK, auto_optimize=optimize)
