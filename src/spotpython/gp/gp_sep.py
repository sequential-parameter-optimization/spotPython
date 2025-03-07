import math
import numpy as np
from spotpython.gp.linalg import linalg_dposv
from spotpython.gp.covar import covar_sep_symm, covar_sep, diff_covar_sep_symm
from spotpython.gp.util import log_determinant_chol
from spotpython.gp.matrix import new_vector, new_id_matrix, new_dup_matrix
from spotpython.gp.lite import predGPsep_lite
from scipy.optimize import minimize
from spotpython.gp.likelihood import nlsep, gradnlsep
import warnings
from typing import Optional, List, Dict, Any, Union
# from scipy.spatial.distance import pdist, squareform
from spotpython.gp.distances import dist


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
        self.gpsepi = 0  # Placeholder if needed for R-interface

    def getDs(self, p: float = 0.1, samp_size: int = 1000) -> dict:
        """
        Calculate a rough starting, minimum, and maximum length-scale from the data X.

        Args:
            p (float): quantile for the distance distribution (default 0.1).
            samp_size (int): sub-sample size if the number of rows in X is large.

        Returns:
            dict: with 'start' (the p-th quantile),
                  'min' (the minimum distance),
                  'max' (the maximum distance).

        Examples:
            >>> from spotpython.gp.gp_sep import GPsep
            >>> import numpy as np
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> gp = GPsep(m=2, n=3, X=X)
            >>> result = gp.getDs(p=0.1, samp_size=10)
            >>> print(result)
        """
        if self.X is None or self.n == 0:
            raise ValueError("The GP model does not have valid data to calculate distances.")

        # Sample rows if needed
        n = self.X.shape[0]
        X_sub = self.X
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

    def darg(self, d, X: np.ndarray = None, samp_size: int = 1000) -> dict:
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
            X = self.X

        if X is None or len(X) == 0:
            raise ValueError("No data found (X is empty).")

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
            Ds = self.getDs(p=0.1, samp_size=samp_size)

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

    def garg(self, g, y: np.ndarray = None) -> dict:
        """
        Process the 'g' argument to set up proper starting values, ranges,
        and priors for the nugget parameter.

        Args:
            g: Could be a dictionary, numeric, or None. If numeric, turn it into {"start": g}.
            y (np.ndarray): The response vector. If None, use self.Z.

        Returns:
            dict: Updated 'g' with fields 'start', 'min', 'max', 'mle', 'ab', etc.
        """
        # If response not provided, default to the GPsep's response
        if y is None:
            y = self.Z

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

    def new_params(self, d: np.ndarray, g: float) -> None:
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

        self.build(dK=True)

    def mleGPsep_both(self, tmin: np.ndarray, tmax: np.ndarray, ab: np.ndarray, maxit: int, verb: int) -> dict:
        """
        Update the separable GP to use its MLE separable lengthscale and multiple nugget parameterization using the current data.

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
        bounds = [(tmin[i], tmax[i]) for i in range(len(p))]
        print(f"bounds: {bounds}")

        def objective(par):
            return nlsep(par, self.X, self.Z)

        def gradient(par):
            return gradnlsep(par, self.X, self.Z)

        result = minimize(fun=objective, x0=p, method="L-BFGS-B", jac=gradient, bounds=bounds, options={"maxiter": maxit, "disp": verb > 0})
        print(f"result: {result}")

        self.d = result.x[:-1]
        self.g = result.x[-1]
        # self.build(dK=False)
        print(f"Updated d: {self.d}, g: {self.g}")

        return {"parameters": result.x, "iterations": result.nit, "convergence": result.status, "message": result.message}

    def mleGPsep_both_R(self, maxit: int, verb: int, tmin: np.ndarray, tmax: np.ndarray, ab: Union[List[float], np.ndarray]) -> dict:
        """
        R-interface to update the separable GP to use its MLE separable lengthscale
        and nugget parameterization using the current data.

        Args:
            maxit (int): Maximum number of iterations.
            verb (int): Verbosity level.
            tmin (np.ndarray): Minimum bounds for the parameters.
            tmax (np.ndarray): Maximum bounds for the parameters.
            ab (List[float] or np.ndarray): Prior parameters (size 4, nonnegative).
        """
        # Convert ab to numpy array if it is a list
        if not isinstance(ab, np.ndarray):
            ab = np.array(ab, dtype=float)

        for j in range(self.m):
            if tmin[j] <= 0:
                tmin[j] = np.finfo(float).eps
            if tmax[j] <= 0:
                tmax[j] = self.m**2
            if self.d[j] > tmax[j]:
                raise ValueError(f"d[{j}]={self.d[j]} > tmax[{j}]={tmax[j]}")
            elif self.d[j] < tmin[j]:
                raise ValueError(f"d[{j}]={self.d[j]} < tmin[{j}]={tmin[j]}")

        if tmin[self.m] <= 0:
            tmin[self.m] = np.finfo(float).eps
        if self.g >= tmax[self.m]:
            raise ValueError(f"g={self.g} >= tmax={tmax[self.m]}")
        elif self.g <= tmin[self.m]:
            raise ValueError(f"g={self.g} <= tmin={tmin[self.m]}")

        # Check for negative entries in ab array
        if np.any(ab < 0):
            raise ValueError("ab must be a positive 4-vector")

        if self.dK is None:
            raise ValueError("derivative info not in GPsep; use newGPsep with dK=True")

        # Call underlying Python or C-based MLE routine
        return self.mleGPsep_both(tmin, tmax, ab, maxit, verb)

    def mleGPsep_main(self, tmin: Optional[List[float]] = None, tmax: Optional[List[float]] = None, ab: Optional[List[float]] = None, maxit: int = 100, verb: int = 0) -> Dict[str, Any]:
        """
        Python version of `mleGPsep_main`, inspired by the R function of the same name.
        This uses L-BFGS-B (through a C call or Python equivalent) to find MLE
        lengthscale (and optionally nugget) for a separable GP.

        Args:
            tmin (List[float], optional): Minimum bounds for d and g.
            tmax (List[float], optional): Maximum bounds for d and g.
            ab (List[float], optional): Hyperparameters for prior (size 4).
            maxit (int): Maximum iterations.
            verb (int): Verbosity level.

        Returns:
            dict: A dictionary with optimized parameters, iteration counts, and more.
        """
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
        if np.any(theta <= tmin):
            print("resetting due to init on lower boundary")
            print(f"theta: {theta}")
            print(f"tmin: {tmin}")
            for i in range(len(tmax)):
                if tmax[i] < 0:
                    tmax[i] = np.sqrt(m)
            theta_new = 0.9 * np.maximum(tmin, 0) + 0.1 * np.array(tmax)
            self.new_params(theta_new[:m], theta_new[m])
            return {
                "theta": theta_new,
                "its": 0,
                "msg": "reset due to init on lower boundary",
                "conv": 102,
            }

        # Call your underlying C/Python L-BFGS-B optimization here; placeholder below:
        out = self.mleGPsep_both_R(maxit=maxit, verb=verb, tmin=tmin, tmax=tmax, ab=ab)

        # After returning, sanity check
        new_theta = np.concatenate((self.get_d(), [self.get_g()]))
        if np.sqrt(np.mean((out["parameters"] - new_theta) ** 2)) > np.sqrt(np.finfo(float).eps):
            warnings.warn("stored theta not the same as theta-hat", RuntimeWarning)

        return {
            "theta": out["parameters"],
            "its": out["iterations"],
            "msg": out["message"],
            "conv": out["convergence"],
        }


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
