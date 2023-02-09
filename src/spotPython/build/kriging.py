import copy
from math import erf
import matplotlib.pyplot as plt
from numpy import min, std, var, mean
from numpy import sqrt
from numpy import exp
from numpy import array
from numpy import log
from numpy import power
from numpy import abs
from numpy import eye
from numpy import sum
from numpy import diag
from numpy import pi
from numpy import ones, zeros
from numpy import spacing, empty_like
from numpy import float64
from numpy import append, ndarray, multiply, isinf, linspace, meshgrid, ravel
from numpy.linalg import cholesky, solve, LinAlgError, cond
from scipy.optimize import differential_evolution
from scipy.linalg import cholesky as scipy_cholesky
import pylab
from spotPython.design.spacefilling import spacefilling
from spotPython.build.surrogates import surrogates
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from spotPython.utils.repair import repair_non_numeric
from spotPython.utils.aggregate import aggregate_mean_var
import logging

logger = logging.getLogger(__name__)
# configure the handler and formatter as needed
py_handler = logging.FileHandler(f"{__name__}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
# add formatter to the handler
py_handler.setFormatter(py_formatter)
# add handler to the logger
logger.addHandler(py_handler)


class Kriging(surrogates):
    def __init__(
            self,
            noise=False,
            cod_type="norm",
            var_type=["num"],
            use_cod_y=False,
            name="kriging",
            seed=124,
            model_optimizer=None,
            model_fun_evals=None,
            min_theta=-3,  # TODO
            max_theta=2,  # TODO
            n_theta=1,
            n_p=1,
            optim_p=False,
            log_level=50,
            **kwargs
    ):
        """
        Kriging surrogate.

        Args:
            noise (bool):
                use regression instead of interpolation kriging. Defaults to "False".
            cod_type (bool):
                normalize or standardize X and values. Can be None, "norm", or "std". Defaults to "norm".
            var_type (str):
                variable type. Can be either `"num`" (numerical) of `"factor"` (factor).
                Defaults to `"num"`.
            use_cod_y (bool):
                use coded y values (instead of natural one). Defaults to `False`.
            name (str):
                Surrogate name. Defaults to `"kriging"`.
            seed (int):
                Random seed. Defaults to `124`.
            model_optimizer (object):
                Optimizer on the surrogate. If `None`, `differential_evolution` is selected.
            model_fun_evals (int):
                Number of iterations used by the optimizer on the surrogate.
            min_theta (float):
                min log10 theta value. Defaults to `-6.`.
            max_theta (float):
                max log10 theta value. Defaults to `3.`.
            n_theta (int):
                number of theta values. Defaults to `1`.
            n_p (int):
                number of p values. Defaults to `1`.
            optim_p (bool):
                Determines whether `p` should be optimized.
            log_level (int):
                logging level, e.g., `20` is `"INFO"`. Defaults to `50` (`"CRITICAL"`).

        Attributes:
            nat_range_X (list):
                List of X natural ranges.
            nat_range_y (list):
                List of y nat ranges.
            noise (bool):
                noisy objective function. Default: False. If `True`, regression kriging will be used.
            var_type (str):
                variable type. Can be either `"num`" (numerical) of `"factor"` (factor).
            num_mask (array):
                array of bool variables. `True` represent numerical (float) variables.
            factor_mask (array):
                array of factor variables. `True` represents factor (unordered) variables.
            int_mask (array):
                array of integer variables. `True` represents integers (ordered) variables.
            ordered_mask (array):
                array of ordered variables. `True` represents integers or float (ordered) variables.
                Set of veriables which an order relation, i.e., they are either num (float) or int.
            name (str):
                Surrogate name
            seed (int):
                Random seed.
            use_cod_y (bool):
                Use coded y values.
            sigma (float):
                Kriging sigma.
            gen (method):
                Design generator, e.g., spotPython.design.spacefilling.spacefilling.
            min_theta (float):
                min log10 theta value. Defaults: -6.
            max_theta (float):
                max log10 theta value. Defaults: 3.
            min_p (float):
                min p value. Default: 1.
            max_p (float):
                max p value. Default: 2.

        Examples:
            Surrogate of the x*sin(x) function.
            See:
            [scikit-learn](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html)

            >>> from spotPython.build.kriging import Kriging
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> rng = np.random.RandomState(1)
            >>> X = linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
            >>> y = np.squeeze(X * np.sin(X))
            >>> training_indices = rng.choice(arange(y.size), size=6, replace=False)
            >>> X_train, y_train = X[training_indices], y[training_indices]
            >>> S = Kriging(name='kriging', seed=124)
            >>> S.fit(X_train, y_train)
            >>> mean_prediction, std_prediction = S.predict(X)
            >>> plt.plot(X, y, label=r"$f(x)$", linestyle="dotted")
            >>> plt.scatter(X_train, y_train, label="Observations")
            >>> plt.plot(X, mean_prediction, label="Mean prediction")
            >>> plt.fill_between(
                X.ravel(),
                mean_prediction - 1.96 * std_prediction,
                mean_prediction + 1.96 * std_prediction,
                alpha=0.5,
                label=r"95% confidence interval",
                )
            >>> plt.legend()
            >>> plt.xlabel("$x$")
            >>> plt.ylabel("$f(x)$")
            >>> _ = plt.title("Gaussian process regression on noise-free dataset")

        """
        super().__init__(name, seed, log_level)

        self.noise = noise
        self.var_type = var_type
        self.cod_type = cod_type
        self.use_cod_y = use_cod_y
        self.name = name
        self.seed = seed
        self.log_level = log_level

        self.sigma = 0
        self.eps = sqrt(spacing(1))
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_p = 1
        self.max_p = 2
        self.min_Lambda = 1e-9
        self.max_Lambda = 1.
        self.n_theta = n_theta
        self.n_p = n_p
        self.optim_p = optim_p
        # Psi matrix condition:
        self.cnd_Psi = 0
        self.inf_Psi = False

        self.model_optimizer = model_optimizer
        if self.model_optimizer is None:
            self.model_optimizer = differential_evolution
        self.model_fun_evals = model_fun_evals
        # differential evaluation uses maxiter = 1000
        # and sets the number of function evaluations to
        # (maxiter + 1) * popsize * N, which results in
        # 1000 * 15 * k, because the default popsize is 15 and
        # N is the number of parameters. This seems to be quite large:
        # for k=2 these are 30 000 iterations. Therefore we set this value to
        # 100
        if self.model_fun_evals is None:
            self.model_fun_evals = 100

        # Logging information
        self.log["negLnLike"] = []
        self.log["theta"] = []
        self.log["p"] = []
        self.log["Lambda"] = []
        # Logger
        logger.setLevel(self.log_level)
        logger.info(f"Starting the logger at level {self.log_level} for module {__name__}:")

    def exp_imp(self, y0, s0):
        """
        Returns the expected improvement for y0 and error s0 (in coded units).
        Args:
            y0 (float):
                function value (in coded units)
            s0 (float):
                error
        Returns:
            (float):
                The expected improvement value.
        """
        # y_min = min(self.cod_y)
        y_min = min(self.mean_cod_y)
        if s0 <= 0.0:
            EI = 0.0
        elif s0 > 0.0:
            EI_one = (y_min - y0) * (
                    0.5 + 0.5 * erf((1.0 / sqrt(2.0)) * ((y_min - y0) / s0))
            )
            EI_two = (s0 * (1.0 / sqrt(2.0 * pi))) * (
                exp(-(1.0 / 2.0) * ((y_min - y0) ** 2.0 / s0 ** 2.0))
            )
            EI = EI_one + EI_two
        return EI

    def set_de_bounds(self):
        """
        Determine search bounds for model_optimizer, e.g., differential evolution.
        """
        de_bounds = []
        for i in range(self.n_theta):
            de_bounds.append([self.min_theta, self.max_theta])
        if self.optim_p:
            for i in range(self.n_p):
                de_bounds.append([self.min_p, self.max_p])
            if self.noise:
                de_bounds.append([self.min_Lambda, self.max_Lambda])
        else:
            if self.noise:
                de_bounds.append([self.min_Lambda, self.max_Lambda])
        self.de_bounds = de_bounds

    def extract_from_bounds(self, new_theta_p_Lambda):
        """
        Extract `theta`, `p`, and `Lambda` from bounds. The kriging object stores
            `theta` as an array,  `p` as an array, and `Lambda` as a float.

        Args:
            new_theta_p_Lambda (numpy.array):
                1d-array with theta, p, and Lambda values. Order is important.

        """
        for i in range(self.n_theta):
            self.theta[i] = new_theta_p_Lambda[i]
        if self.optim_p:
            for i in range(self.n_p):
                self.p[i] = new_theta_p_Lambda[i + self.n_theta]
            if self.noise:
                self.Lambda = new_theta_p_Lambda[self.n_theta + self.n_p]
        else:
            if self.noise:
                self.Lambda = new_theta_p_Lambda[self.n_theta]

    def fit(self, nat_X, nat_y):
        """
        The function fits the hyperparameters (`theta`, `p`, `Lambda`) of the Kriging model, i.e.,
        the following internal values are computed:

        1. `theta`, `p`, and `Lambda` values via optimization of the function `fun_likelihood()`.
        2. Correlation matrix `Psi` via `rebuildPsi()`.

        Args:
            nat_X (array):
                sample points
            nat_y (array):
                function values

        Returns:
            surrogate (object):
                Fitted estimator.

        Attributes:
            theta (numpy.ndarray):
                Kriging theta values. Shape (k,).
            p (numpy.ndarray):
                Kriging p values. Shape (k,).
            LnDetPsi (numpy.float64):
                Determinant Psi matrix.
            Psi (numpy.matrix):
                Correlation matrix Psi. Shape (n,n).
            psi (numpy.ndarray):
                psi vector. Shape (n,).
            one (numpy.ndarray):
                vector of ones. Shape (n,).
            mu (numpy.float64):
                Kriging expected mean value mu.
            U (numpy.matrix):
                Kriging U matrix, Cholesky decomposition. Shape (n,n).
            SigmaSqr (numpy.float64):
                Sigma squared value.
            Lambda (float):
                lambda noise value.

        """
        self.nat_X = copy.deepcopy(nat_X)
        self.nat_y = copy.deepcopy(nat_y)
        self.n = self.nat_X.shape[0]
        self.k = self.nat_X.shape[1]
        self.cod_X = empty_like(self.nat_X)
        self.cod_y = empty_like(self.nat_y)

        # assume all variable types are "num" if "num" is
        # specified once:
        if len(self.var_type) < self.k:
            self.var_type = self.var_type * self.k
            logger.warning("Warning: All variable types forced to 'num'.")
        self.num_mask = array(list(map(lambda x: x == "num", self.var_type)))
        self.factor_mask = array(list(map(lambda x: x == "factor", self.var_type)))
        self.int_mask = array(list(map(lambda x: x == "int", self.var_type)))
        self.ordered_mask = array(list(map(lambda x: x == "int" or x == "num", self.var_type)))
        self.nat_to_cod_init()
        if self.n_theta > self.k:
            self.n_theta = self.k
            logger.warning("More theta values than dimensions. `n_theta` set to `k`.")
        self.theta = zeros(self.n_theta)
        # TODO: Currently not used:
        self.x0_theta = ones((self.n_theta,)) * self.n / (100 * self.k)
        self.p = ones(self.n_p) * 2.0

        self.pen_val = self.n * log(var(self.nat_y)) + 1e4
        self.negLnLike = None

        self.gen = spacefilling(k=self.k, seed=self.seed)

        # matrix related
        self.LnDetPsi = None
        self.Psi = zeros((self.n, self.n), dtype=float64)
        self.psi = zeros((self.n, 1))
        self.one = ones(self.n)
        self.mu = None
        self.U = None
        self.SigmaSqr = None
        self.Lambda = None
        # build_Psi() and build_U() are called in fun_likelihood
        self.set_de_bounds()
        if self.model_optimizer.__name__ == 'dual_annealing':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          bounds=self.de_bounds)
        elif self.model_optimizer.__name__ == 'differential_evolution':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          bounds=self.de_bounds,
                                          maxiter=self.model_fun_evals,
                                          seed=self.seed)
        elif self.model_optimizer.__name__ == 'direct':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          bounds=self.de_bounds,
                                          # maxfun=self.model_fun_evals,
                                          eps=1e-2)
        elif self.model_optimizer.__name__ == 'shgo':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          bounds=self.de_bounds)
        elif self.model_optimizer.__name__ == 'basinhopping':
            result = self.model_optimizer(func=self.fun_likelihood,
                                          x0=mean(self.de_bounds, axis=1))
        else:
            result = self.model_optimizer(func=self.fun_likelihood, bounds=self.de_bounds)
        # Finally, set new theta and p values and update the surrogate again
        # for new_theta_p_Lambda in de_results["x"]:
        new_theta_p_Lambda = result["x"]
        self.extract_from_bounds(new_theta_p_Lambda)
        self.build_Psi()
        self.build_U()
        # TODO: check if the following line is necessary!
        self.likelihood()
        self.log["negLnLike"] = append(self.log["negLnLike"], self.negLnLike)
        self.log["theta"] = append(self.log["theta"], self.theta)
        self.log["p"] = append(self.log["p"], self.p)
        self.log["Lambda"] = append(self.log["Lambda"], self.Lambda)
        # TODO: return self

    def fun_likelihood(self, new_theta_p_Lambda):
        """
        Compute log likelihood for a set of hyperparameters (theta, p, Lambda).
        Performs the following steps:

        1. Build Psi via `build_Psi()` and `build_U()`.
        2. Compute negLnLikelihood via `likelihood()
        3. If successful, the return `negLnLike` value, otherwise a penalty value (`pen_val`).

        Args:
            new_theta_p_Lambda (array):
                `theta`, `p`, and `Lambda` values stored in an array.
        Returns:
            (float):
                negLnLike, th negative log likelihood of the surface at the hyperparameters specified.
        """
        self.extract_from_bounds(new_theta_p_Lambda)
        if self.__is_any__(power(10.0, self.theta), 0):
            # print(f"Failure in fun_likelihood: 10^theta == 0. Setting negLnLike to {self.pen_val:.2f}.")
            logger.warning("Failure in fun_likelihood: 10^theta == 0. Setting negLnLike to %s", self.pen_val)
            return self.pen_val
        self.build_Psi()
        if (self.inf_Psi or self.cnd_Psi > 1e9):
            # print(f"\nFailure in fun_likelihood: Psi is ill conditioned ({self.cnd_Psi}).")
            # print(f"Setting negLnLike to {self.pen_val:.2f}.")
            logger.warning("Failure in fun_likelihood: Psi is ill conditioned: %s", self.cnd_Psi)
            logger.warning("Setting negLnLike to: %s", self.pen_val)
            return self.pen_val
        else:
            try:
                self.build_U()
            except Exception as err:
                f = self.pen_val
                print(f"Error in fun_likelihood(). Call to build_U() failed. {err=}, {type(err)=}")
                print(f"Setting negLnLike to {self.pen_val:.2f}.")
                return f
            self.likelihood()
            return self.negLnLike

    def __is_any__(self, x, v):
        if not isinstance(x, ndarray):
            x = array([x])
        return any(x == v)

    def build_Psi(self):
        """
        New construction (rebuild to reflect new data or a change in hyperparameters)
        of the (nxn) correlation matrix Psi as described in [Forr08a, p.57].
        Note:
            Method uses `theta`, `p`, and coded `X` values.
        """
        self.Psi = zeros((self.n, self.n), dtype=float64)
        theta = power(10.0, self.theta)
        if self.n_theta == 1:
            theta = theta * ones(self.k)
        try:
            D = zeros((self.n, self.n))
            if self.ordered_mask.any():
                X_ordered = self.cod_X[:, self.ordered_mask]
                D = squareform(
                    pdist(
                        X_ordered, metric='sqeuclidean', out=None, w=theta[self.ordered_mask]))
            if self.factor_mask.any():
                X_factor = self.cod_X[:, self.factor_mask]
                D = (D + squareform(
                    pdist(X_factor,
                          metric='hamming',
                          out=None,
                          w=theta[self.factor_mask])))
            self.Psi = exp(-D)
        except LinAlgError as err:
            print(f"Building Psi failed:\n {self.Psi}. {err=}, {type(err)=}")
        if self.noise:
            self.Psi = self.Psi + multiply(eye(self.n), self.Lambda)
        else:
            self.Psi = self.Psi + multiply(eye(self.n), self.eps)
        if (isinf(self.Psi)).any():
            self.inf_Psi = True
        self.cnd_Psi = cond(self.Psi)

    def build_U(self, scipy=True):
        """
        Cholesky factorization of Psi as U as described in [Forr08a, p.57].

        Args:
            scipy (bool): Use `scipy_cholesky`. If `False`, numpy's `cholesky` is used.
        """
        try:
            if scipy:
                self.U = scipy_cholesky(self.Psi, lower=True)
            else:
                self.U = cholesky(self.Psi)
            self.U = self.U.T
        except LinAlgError as err:
            print(f"build_U() Cholesky failed for Psi:\n {self.Psi}. {err=}, {type(err)=}")

    def likelihood(self):
        """
        Calculates the negative of the concentrated log-likelihood.
        Implementation of (2.32)  in [Forr08a].
        See also function krigingLikelihood() in spot.
        Note:
            `build_Psi` and `build_U` should be called first.
        Modifies:
            `mu`,
            `SigmaSqr`,
            `LnDetPsi`, and
            `negLnLike`, concentrated log-likelihood *-1 for minimizing

        """
        # (2.20) in [Forr08a]:
        mu = (
                 self.one.T.dot(
                     solve(self.U, solve(self.U.T, self.cod_y))
                 )
             ) / self.one.T.dot(solve(self.U, solve(self.U.T, self.one)))
        self.mu = mu
        # (2.31) in [Forr08a]
        self.SigmaSqr = (
                            (self.cod_y - self.one.dot(self.mu)).T.dot(
                                solve(
                                    self.U,
                                    solve(self.U.T, (self.cod_y - self.one.dot(self.mu))),
                                )
                            )
                        ) / self.n
        # (2.32) in [Forr08a]
        self.LnDetPsi = 2.0 * sum(log(abs(diag(self.U))))
        self.negLnLike = -1.0 * (-(self.n / 2.0) * log(self.SigmaSqr) - 0.5 * self.LnDetPsi)

    def plot(self, show=True):
        """
        This function plots 1d and 2d surrogates.
        Args:
            show (boolean):
                If `True`, the plots are displayed.
                If `False`, `plt.show()` should be called outside this function.
        """
        if self.k == 1:
            # TODO: Improve plot (add conf. interval etc.)
            fig = pylab.figure(figsize=(9, 6))
            # t1 = array(arange(0.0, 1.0, 0.01))
            # y1 = array([self.predict(array([x]), return_val="y") for x in t1])
            # plt.figure()
            # plt.plot(t1, y1, "k")
            # if show:
            #     plt.show()
            #
            n_grid = 100
            x = linspace(
                self.nat_range_X[0][0], self.nat_range_X[0][1], num=n_grid
            )
            y = self.predict(x)
            plt.figure()
            plt.plot(x, y, "k")
            if show:
                plt.show()

        if self.k == 2:
            fig = pylab.figure(figsize=(9, 6))
            n_grid = 100
            x = linspace(
                self.nat_range_X[0][0], self.nat_range_X[0][1], num=n_grid
            )
            y = linspace(
                self.nat_range_X[1][0], self.nat_range_X[1][1], num=n_grid
            )
            X, Y = meshgrid(x, y)
            # Predict based on the optimized results
            zz = array(
                [self.predict(array([x, y]), return_val="all") for x, y in zip(ravel(X), ravel(Y))]
            )
            zs = zz[:, 0, :]
            zse = zz[:, 1, :]
            Z = zs.reshape(X.shape)
            Ze = zse.reshape(X.shape)

            if self.cod_type == "norm":
                nat_point_X = (
                                      self.cod_X[:, 0] * (self.nat_range_X[0][1] - self.nat_range_X[0][0])
                              ) + self.nat_range_X[0][0]
                nat_point_Y = (
                                      self.cod_X[:, 1] * (self.nat_range_X[1][1] - self.nat_range_X[1][0])
                              ) + self.nat_range_X[1][0]
            elif self.cod_type == "std":
                nat_point_X = self.cod_X[:, 0] * self.nat_std_X[0] + self.nat_mean_X[0]
                nat_point_Y = self.cod_X[:, 1] * self.nat_std_X[1] + self.nat_mean_X[1]
            else:
                nat_point_X = self.cod_X[:, 0]
                nat_point_Y = self.cod_X[:, 1]
            contour_levels = 30
            ax = fig.add_subplot(224)
            # plot predicted values:
            pylab.contourf(X, Y, Ze, contour_levels, cmap="jet")
            pylab.title("Error")
            pylab.colorbar()
            # plot observed points:
            pylab.plot(nat_point_X, nat_point_Y, "ow")
            #
            ax = fig.add_subplot(223)
            # plot predicted values:
            plt.contourf(X, Y, Z, contour_levels, zorder=1, cmap="jet")
            plt.title("Surrogate")
            # plot observed points:
            pylab.plot(nat_point_X, nat_point_Y, "ow", zorder=3)
            pylab.colorbar()
            #
            ax = fig.add_subplot(221, projection="3d")
            ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.9, cmap="jet")
            #
            ax = fig.add_subplot(222, projection="3d")
            ax.plot_surface(X, Y, Ze, rstride=3, cstride=3, alpha=0.9, cmap="jet")
            #
            pylab.show()

    def predict(self, nat_X, nat=True, return_val="y"):
        """
        This function returns the prediction (in natural units) of the surrogate
        at the natural coordinates of X.

        Args:
            nat_X (array):
                Design variable to evaluate in natural units.
            nat (bool):
                argument `nat_X` is in natural range. Default: `True`. If set to `False`,
                `nat_X` will not be normalized (which might be useful if already normalized
                y values are used).
            return_val (string): whether `y`, `s`, neg. `ei` (negative expected improvement),
                or all three values are returned. Default is (for compatibility with sklearn) "y".
                To return `s`, select "s", to return neg. `ei`, select "ei".
                To return the tuple `(y, s, ei)`, select "all".
        Returns:
            (float):
                The predicted value in natural units.
            (float):
                predicted error
            (float):
                expected improvement
        """
        # Check for the shape and the type of the Input
        if isinstance(nat_X, ndarray):
            try:
                X = nat_X.reshape(-1, self.nat_X.shape[1])
                X = repair_non_numeric(X, self.var_type)
            except Exception:
                raise TypeError("13.1: Input to predict was not convertible to the size of X")
        else:
            raise TypeError(f"type of the given input is an {type(nat_X)} instead of an ndarray")

        # Iterate through the Input
        y = array([], dtype=float)
        s = array([], dtype=float)
        ei = array([], dtype=float)

        for i in range(X.shape[0]):
            # logger.debug(f"13.2: predict() Step 2: x (reshaped nat_X):\n {x}")
            if nat:
                x = self.nat_to_cod_x(X[i, :])
            else:
                x = X[i, :]
            y0, s0, ei0 = self.predict_coded(x)
            y = append(y, y0)
            s = append(s, s0)
            ei = append(ei, ei0)
        if return_val == "y":
            return y
        elif return_val == "s":
            return s
        elif return_val == "ei":
            return -1.0 * ei
        else:
            return y, s, -1.0 * ei

    def build_psi_vec(self, cod_x):
        """
        Build the psi vector. Needed by `predict_cod`, `predict_err_coded`,
        `regression_predict_coded`. Modifies `self.psi`.

        Args:
            cod_x (array): point to calculate psi

        """
        self.psi = zeros((self.n))
        # theta = self.theta  # TODO:
        theta = power(10.0, self.theta)
        if self.n_theta == 1:
            theta = theta * ones(self.k)
        try:
            D = zeros((self.n))
            if self.ordered_mask.any():
                X_ordered = self.cod_X[:, self.ordered_mask]
                x_ordered = cod_x[self.ordered_mask]
                D = cdist(x_ordered.reshape(-1, sum(self.ordered_mask)),
                          X_ordered.reshape(-1, sum(self.ordered_mask)),
                          metric='sqeuclidean',
                          out=None,
                          w=theta[self.ordered_mask])
            if self.factor_mask.any():
                X_factor = self.cod_X[:, self.factor_mask]
                x_factor = cod_x[self.factor_mask]
                D = (D + cdist(x_factor.reshape(-1, sum(self.factor_mask)),
                               X_factor.reshape(-1, sum(self.factor_mask)),
                               metric='hamming',
                               out=None,
                               w=theta[self.factor_mask]))
            self.psi = exp(-D).T
        except LinAlgError as err:
            print(f"Building psi failed:\n {self.psi}. {err=}, {type(err)=}")

    def predict_coded(self, cod_x):
        """
        Kriging prediction of one point in the coded units as described in (2.20) in [Forr08a].
        The error is returned as well.
        See also [Forr08a, p.60].
        Note:
            `self.mu` and `self.SigmaSqr` are computed in `likelihood`, not here.
        Args:
           cod_x (array):
            point in coded units to make prediction at
        Returns:
            (float):
                predicted value in coded units.
            (float):
                predicted error.
        """
        self.build_psi_vec(cod_x)
        f = self.mu + self.psi.T.dot(
            solve(self.U, solve(
                self.U.T, self.cod_y - self.one.dot(self.mu)))
        )
        try:
            if self.noise:
                Lambda = self.Lambda
            else:
                Lambda = 0.0
            # Error in [Forr08a, p.87]:
            SSqr = self.SigmaSqr * (1 + Lambda - self.psi.T.dot(solve(self.U, solve(self.U.T, self.psi))))
        except Exception as err:
            print(f"Could not determine SSqr. Wrong or missing Lambda? {err=}, {type(err)=}")

        SSqr = power(abs(SSqr[0]), 0.5)[0]
        EI = self.exp_imp(y0=f[0], s0=SSqr)
        return f[0], SSqr, EI

    def weighted_exp_imp(self, cod_x, w):
        """
        Weighted expected improvement.

        References:
            [Sobester et al. 2005].

        Args:
            cod_x (array):
                A coded design vector.
            w (float):
                weight

        Returns:
            (float):
                weighted expected improvement.
        """
        y0, s0 = self.predict_coded(cod_x)
        y_min = min(self.cod_y)
        if s0 <= 0.0:
            EI = 0.0
        elif s0 > 0.0:
            EI_one = w * (
                    (y_min - y0)
                    * (0.5 + 0.5 * erf((1.0 / sqrt(2.0)) * ((y_min - y0) / s0)))
            )
            EI_two = (
                    (1.0 - w)
                    * (s0 * (1.0 / sqrt(2.0 * pi)))
                    * (exp(-(1.0 / 2.0) * ((y_min - y0) ** 2.0 / s0 ** 2.0)))
            )
            EI = EI_one + EI_two
        return EI

    def calculate_mean_MSE(self, n_samples=200, points=None):
        """
        This function calculates the mean MSE metric of the model by evaluating MSE at a number of points.
        Args:
            n_samples (integer):
                Number of points to sample the mean squared error at.
                Ignored if the points argument is specified.
            points (array):
                an array of points to sample the model at.
        Returns:
            (float):
                the mean value of MSE and the standard deviation of the MSE points
        """
        if points is None:
            points = self.gen.lhd(n_samples)
        values = zeros(len(points))
        for enu, point in enumerate(points):
            s0 = self.predict(cod_X=point, nat=True, return_val="s")
            values[enu] = s0
        return mean(values), std(values)

    def cod_to_nat_x(self, cod_X):
        """
        Args:
            cod_X (array):
                An array representing one point (self.k long) in normalized (coded) units.
        Returns:
            (array):
                An array of natural (physical or real world) units.
        """
        X = copy.deepcopy(cod_X)
        if self.cod_type == "norm":
            for i in range(self.k):
                X[i] = (
                               X[i] * float(self.nat_range_X[i][1] - self.nat_range_X[i][0])
                       ) + self.nat_range_X[i][0]
            return X
        elif self.cod_type == "std":
            for i in range(self.k):
                X[i] = X[i] * self.nat_std_X[i] + self.nat_mean_X[i]
            return X
        else:
            return cod_X

    def cod_to_nat_y(self, cod_y):
        """
        Args:
            cod_y (array):
                A normalized array of coded (model) units in the range of [0,1].
        Returns:
            (array):
                An array of observed values in real-world units.
        """
        if self.cod_type == "norm":
            return (
                           cod_y * (self.nat_range_y[1] - self.nat_range_y[0])
                   ) + self.nat_range_y[0]
        elif self.cod_type == "std":
            return cod_y * self.nat_std_y + self.nat_mean_y
        else:
            return cod_y

    def nat_to_cod_x(self, nat_X):
        """
        Normalize one point (row) of nat_X array to [0,1]. The internal nat_range_X values are not updated.

        Args:
            nat_X (array):
                An array representing one points (self.k long) in natural (physical or real world) units.
        Returns:
            (array):
                An array of coded values in the range of [0,1] for each dimension.
        """
        X = copy.deepcopy(nat_X)
        if self.cod_type == "norm":
            for i in range(self.k):
                # TODO: Check Implementation of range correction if range == 0:
                # rangex <- xmax - xmin
                # rangey <- ymax - ymin
                # xmin[rangex == 0] <- xmin[rangex == 0] - 0.5
                # xmax[rangex == 0] <- xmax[rangex == 0] + 0.5
                # rangex[rangex == 0] <- 1
                # logger.debug(f"self.nat_range_X[{i}]:\n {self.nat_range_X[i]}")
                # logger.debug(f"X[{i}]:\n {X[i]}")
                rangex = float(self.nat_range_X[i][1] - self.nat_range_X[i][0])
                if rangex == 0:
                    self.nat_range_X[i][0] = self.nat_range_X[i][0] - 0.5
                    self.nat_range_X[i][1] = self.nat_range_X[i][1] + 0.5
                X[i] = (X[i] - self.nat_range_X[i][0]) / float(self.nat_range_X[i][1] - self.nat_range_X[i][0])
            return X
        elif self.cod_type == "std":
            for i in range(self.k):
                X[i] = (X[i] - self.nat_mean_X[i]) / self.nat_std_X[i]
            return X
        else:
            return nat_X

    def nat_to_cod_y(self, nat_y):
        """
        Normalize natural y values to [0,1].
        Args:
            nat_y (array):
                An array of observed values in natural (real-world) units.
        Returns:
            (array):
                A normalized array of coded (model) units in the range of [0,1].

        """
        if self.use_cod_y:
            if self.cod_type == "norm":
                return (nat_y - self.nat_range_y[0]) / (self.nat_range_y[1] - self.nat_range_y[0])
            elif self.cod_type == "std":
                return (nat_y - self.nat_mean_y) / self.nat_std_y
            else:
                return nat_y
        else:
            return nat_y

    def nat_to_cod_init(self):
        """
        Determine max and min of each dimension and normalize that axis to a range of [0,1].
        Called when 1) surrogate is initialized and 2) new points arrive, i.e., suggested
        by the surrogate as infill points.
        This method calls `nat_to_cod_x` and `nat_to_cod_y` and updates the ranges `nat_range_X` and
        `nat_range_y`.
        """
        self.nat_range_X = []
        self.nat_range_y = []
        for i in range(self.k):
            self.nat_range_X.append([min(self.nat_X[:, i]), max(self.nat_X[:, i])])
        self.nat_range_y.append(min(self.nat_y))
        self.nat_range_y.append(max(self.nat_y))
        self.nat_mean_X = mean(self.nat_X, axis=0)
        self.nat_std_X = std(self.nat_X, axis=0)
        self.nat_mean_y = mean(self.nat_y)
        self.nat_std_y = std(self.nat_y)
        Z = aggregate_mean_var(X=self.nat_X, y=self.nat_y)
        mu = Z[1]
        self.mean_cod_y = empty_like(mu)

        for i in range(self.n):
            self.cod_X[i] = self.nat_to_cod_x(self.nat_X[i])
        for i in range(self.n):
            self.cod_y[i] = self.nat_to_cod_y(self.nat_y[i])
        for i in range(mu.shape[0]):
            self.mean_cod_y[i] = self.nat_to_cod_y(mu[i])
