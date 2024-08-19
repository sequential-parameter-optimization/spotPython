import numpy as np
from numpy.random import default_rng
from random import random
from typing import List, Optional, Dict


class analytical:
    """
    Class for analytical test functions.

    Args:
        offset (float):
            Offset value. Defaults to 0.0.
        hz (float):
            Horizontal value. Defaults to 0.
        seed (int):
            Seed value for random number generation. Defaults to 126.

    Notes:
        See [Numpy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start)

    Attributes:
        offset (float):
            Offset value.
        hz (float):
            Horizontal value.
        sigma (float):
            Noise level.
        seed (int):
            Seed value for random number generation.
        rng (Generator):
            Numpy random number generator object.
        fun_control (dict):
            Dictionary containing control parameters for the function.
    """

    def __init__(self, offset: float = 0.0, hz: float = 0, sigma=0.0, seed: int = 126) -> None:
        self.offset = offset
        self.hz = hz
        self.sigma = sigma
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.fun_control = {"sigma": sigma, "seed": None, "sel_var": None}

    def __repr__(self) -> str:
        return f"analytical(offset={self.offset}, hz={self.hz}, seed={self.seed})"

    def add_noise(self, y: List[float]) -> np.ndarray:
        """
        Adds noise to the input data.
        This method takes in a list of float values y as input and adds noise to
        the data using a random number generator. The method returns a numpy array
        containing the noisy data.

        Args:
            self (analytical): analytical class object.
            y (List[float]): Input data.

        Returns:
            np.ndarray: Noisy data.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
                import numpy as np
                y = np.array([1, 2, 3, 4, 5])
                fun = analytical(sigma=1.0, seed=123)
                fun.add_noise(y)
            array([0.01087865, 1.63221335, 4.28792526, 4.19397442, 5.9202309 ])

        """
        # Use own rng:
        if self.fun_control["seed"] is not None:
            rng = default_rng(seed=self.fun_control["seed"])
        # Use class rng:
        else:
            rng = self.rng
        noise_y = np.array([], dtype=float)
        for y_i in y:
            noise_y = np.append(
                noise_y,
                y_i + rng.normal(loc=0, scale=self.fun_control["sigma"], size=1),
            )
        return noise_y

    def fun_branin_factor(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """
        Calculates the Branin function of (x1, x2) with an additional factor based on the value of x3.
        If x3 = 1, the value of the Branin function is increased by 10.
        If x3 = 2, the value of the Branin function is decreased by 10.
        Otherwise, the value of the Branin function is not changed.

        Args:
            X (np.ndarray):
                A 2D numpy array with shape (n, 3) where n is the number of samples.
            fun_control (Optional[Dict]):
                A dictionary containing control parameters for the function.
                If None, self.fun_control is used. Defaults to None.

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
                import numpy as np
                X = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
                fun = analytical()
                fun.fun_branin_factor(X)
                array([55.60211264, 65.60211264, 45.60211264])
        """
        if fun_control is None:
            fun_control = self.fun_control
        if len(X.shape) == 1:
            X = np.array([X])
        if X.shape[1] != 3:
            raise Exception("X must have shape (n, 3)")
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        for j in range(X.shape[0]):
            if x3[j] == 1:
                y[j] = y[j] + 10
            elif x3[j] == 2:
                y[j] = y[j] - 10
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_linear(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Linear function.

        Args:
            X (array):
                input
            fun_control (dict):
                dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> fun = analytical()
            >>> fun.fun_linear(X)
            array([ 6., 15.])

        """
        if fun_control is not None:
            self.fun_control = fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = np.atleast_2d(X)
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, np.sum(X[i]))
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_sphere(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Sphere function.

        Args:
            X (array):
                input
            fun_control (dict):
                dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> fun = analytical()
            >>> fun.fun_sphere(X)
            array([14., 77.])

        """
        if fun_control is not None:
            self.fun_control = fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = np.atleast_2d(X)
        offset = np.ones(X.shape[1]) * self.offset
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, np.sum((X[i] - offset) ** 2))
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_cubed(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Cubed function.

        Args:
            X (array):
                input
            fun_control (dict):
                dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> fun = analytical()
            >>> fun.fun_cubed(X)
            array([ 0., 27.])
        """

        if fun_control is None:
            fun_control = self.fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X = np.atleast_2d(X)
        offset = np.ones(X.shape[1]) * self.offset
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, np.sum((X[i] - offset) ** 3))
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_forrester(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Forrester function. Function used by [Forr08a, p.83].
           f(x) = (6x- 2)^2 sin(12x-4) for x in [0,1].
           Starts with three sample points at x=0, x=0.5, and x=1.

        Args:
            X (array):
                input
            fun_control (dict):
                dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> fun = analytical()
            >>> fun.fun_forrester(X)
            array([  0.        ,  11.99999999])
        """
        if fun_control is None:
            fun_control = self.fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X = np.atleast_2d(X)
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, (6.0 * X[i] - 2) ** 2 * np.sin(12 * X[i] - 4))
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_branin(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        r"""Branin function. The 2-dim Branin function is defined as
            $$
            y = a (x_2 - b x_1^2 + c x_1 - r) ^2 + s (1 - t) \cos(x_1) + s,
            $$
            where values of $a, b, c, r, s$ and $t$ are:
            $a = 1$, $b = 5.1 / (4\pi^2)$, $c = 5 / \pi$, $r = 6$, $s = 10$ and $t = 1 / (8\pi)$.
            It has three global minima with $f(x) = 0.39788736$ at
            $$
            (-\pi, 12.275),
            $$
            $$
            (\pi, 2.275),
            $$
            and
            $$
            (9.42478, 2.475).
            $$
            Input domain: This function is usually evaluated on the square $x_1 \in [-5, 10] \times x_2 \in [0, 15]$.

        Args:
            X (array):
                input
            fun_control (dict):
                dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
                pi = np.pi
                X = np.array([[0,0],
                    [-pi, 12.275],
                    [pi, 2.275],
                    [9.42478, 2.475]])
                fun = analytical()
                fun.fun_branin(X)
                array([55.60211264,  0.39788736,  0.39788736,  0.39788736])

        """
        if fun_control is None:
            fun_control = self.fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.shape[1] != 2:
            raise Exception
        x1 = X[:, 0]
        x2 = X[:, 1]
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_branin_modified(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Modified Branin function.

        Args:
            X (array):
                input
            fun_control (dict):
                dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> fun = analytical()
            >>> fun.fun_branin_modified(X)
            array([  0.        ,  11.99999999])

        """
        if fun_control is None:
            fun_control = self.fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.shape[1] != 2:
            raise Exception
        x = X[:, 0]
        y = X[:, 1]
        X1 = 15 * x - 5
        X2 = 15 * y
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        d = 6
        e = 10
        ff = 1 / (8 * np.pi)
        y = (a * (X2 - b * X1**2 + c * X1 - d) ** 2 + e * (1 - ff) * np.cos(X1) + e) + 5 * x
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def branin_noise(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Branin function with noise.

        Args:
            X (array):
                input
            fun_control (dict):
                dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            (np.ndarray): A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> fun = analytical()
            >>> fun.branin_noise(X)
            array([  0.        ,  11.99999999])

        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.shape[1] != 2:
            raise Exception
        x = X[:, 0]
        y = X[:, 1]
        X1 = 15 * x - 5
        X2 = 15 * y
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        d = 6
        e = 10
        ff = 1 / (8 * np.pi)
        noiseFree = (a * (X2 - b * X1**2 + c * X1 - d) ** 2 + e * (1 - ff) * np.cos(X1) + e) + 5 * x
        noise_y = []
        for i in noiseFree:
            noise_y.append(i + np.random.standard_normal() * 15)
        return np.array(noise_y)

    def fun_sin_cos(self, X, fun_control=None):
        """Sinusoidal function.
        Args:
            X (array):
                input
            fun_control (dict):
                dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            (np.ndarray): A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> fun = analytical()
            >>> fun.fun_sin_cos(X)
            array([-1.        , -0.41614684])
        """

        if fun_control is None:
            fun_control = self.fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.shape[1] != 2:
            raise Exception
        x0 = X[:, 0]
        x1 = X[:, 1]
        y = 2.0 * np.sin(x0 + self.hz) + 0.5 * np.cos(x1 + self.hz)
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    # def fun_forrester_2(self, X):
    #     """
    #     Function used by [Forr08a, p.83].
    #     f(x) = (6x- 2)^2 sin(12x-4) for x in [0,1].
    #     Starts with three sample points at x=0, x=0.5, and x=1.

    #     Args:
    #         X (flooat): input values (1-dim)

    #     Returns:
    #         float: function value
    #     """
    #     try:
    #         X.shape[1]
    #     except ValueError:
    #         X = np.array(X)

    # X = np.atleast_2d(X)
    #     # y = X[:, 1]
    #     y = (6.0 * X - 2) ** 2 * np.sin(12 * X - 4)
    #     if self.sigma != 0:
    #         noise_y = np.array([], dtype=float)
    #         for i in y:
    #             noise_y = np.append(
    #                 noise_y, i + np.random.normal(loc=0, scale=self.sigma, size=1)
    #             )
    #         return noise_y
    #     else:
    #         return y

    def fun_runge(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Runge function. Formula: f(x) = 1/ (1 + sum(x_i) - offset)^2. Dim: k >= 1.
           Interval: -5 <= x <= 5

        Args:
            X (array): input
            fun_control (dict): dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> fun = analytical()
            >>> fun.fun_runge(X)
            array([0.0625    , 0.015625  , 0.00390625])

        """
        if fun_control is None:
            fun_control = self.fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        offset = np.ones(X.shape[1]) * self.offset
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, (1 / (1 + np.sum((X[i] - offset) ** 2))))
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_wingwt(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        r"""Wing weight function. Example from Forrester et al. to understand the weight
            of an unpainted light aircraft wing as a function of nine design and operational parameters:
            $W=0.036 S_W^{0.758}  Wfw^{0.0035} ( A / (\cos^2 \Lambda))^{0.6} q^{0.006}  \lambda^{0.04} ( (100 Rtc)/(\cos
              \Lambda) ))^{-0.3} (Nz Wdg)^{0.49}$

        | Symbol    | Parameter                              | Baseline | Minimum | Maximum |
        |-----------|----------------------------------------|----------|---------|---------|
        | $S_W$     | Wing area ($ft^2$)                     | 174      | 150     | 200     |
        | $W_{fw}$  | Weight of fuel in wing (lb)            | 252      | 220     | 300     |
        | $A$       | Aspect ratio                          | 7.52     | 6       | 10      |
        | $\Lambda$ | Quarter-chord sweep (deg)              | 0        | -10     | 10      |
        | $q$       | Dynamic pressure at cruise ($lb/ft^2$) | 34       | 16      | 45      |
        | $\lambda$ | Taper ratio                            | 0.672    | 0.5     | 1       |
        | $R_{tc}$  | Aerofoil thickness to chord ratio      | 0.12     | 0.08    | 0.18    |
        | $N_z$     | Ultimate load factor                   | 3.8      | 2.5     | 6       |
        | $W_{dg}$  | Flight design gross weight (lb)         | 2000     | 1700    | 2500    |
        | $W_p$     | paint weight (lb/ft^2)                   | 0.064 |   0.025  | 0.08    |

        Args:
            X (array): input
            fun_control (dict): dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 10, 11, 12]])
            >>> fun = analytical()
            >>> fun.fun_wingwt(X)
            array([0.0625    , 0.015625  , 0.00390625])

        """
        if fun_control is None:
            fun_control = self.fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        #
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            Sw = X[i, 0] * (200 - 150) + 150
            Wfw = X[i, 1] * (300 - 220) + 220
            A = X[i, 2] * (10 - 6) + 6
            L = (X[i, 3] * (10 - (-10)) - 10) * np.pi / 180
            q = X[i, 4] * (45 - 16) + 16
            la = X[i, 5] * (1 - 0.5) + 0.5
            Rtc = X[i, 6] * (0.18 - 0.08) + 0.08
            Nz = X[i, 7] * (6 - 2.5) + 2.5
            Wdg = X[i, 8] * (2500 - 1700) + 1700
            Wp = X[i, 9] * (0.08 - 0.025) + 0.025
            # calculation on natural scale
            W = 0.036 * Sw**0.758 * Wfw**0.0035 * (A / np.cos(L) ** 2) ** 0.6 * q**0.006
            W = W * la**0.04 * (100 * Rtc / np.cos(L)) ** (-0.3) * (Nz * Wdg) ** (0.49) + Sw * Wp
            y = np.append(y, W)
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_xsin(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Example function.
        Args:
            X (array): input
            fun_control (dict): dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 10, 11, 12]])
            >>> fun = analytical()
            >>> fun.fun_xsin(X)
            array([0.84147098, 0.90929743, 0.14112001])

        """
        if fun_control is None:
            fun_control = self.fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = np.atleast_2d(X)
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, X[i] * np.sin(1.0 / X[i]))
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_rosen(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Rosenbrock function.
        Args:
            X (array): input
            fun_control (dict): dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2,], [4, 5 ]])
            >>> fun = analytical()
            >>> fun.fun_rosen(X)
            array([24,  0])
        """

        if fun_control is None:
            fun_control = self.fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.shape[1] != 2:
            raise Exception
        x0 = X[:, 0]
        x1 = X[:, 1]
        b = 10
        y = (x0 - 1) ** 2 + b * (x1 - x0**2) ** 2
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_random_error(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Return errors for testing spot stability.
        Args:
            X (array): input
            fun_control (dict): dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([[1, 2,], [4, 5 ]])
            >>> fun = analytical()
            >>> fun.fun_random_error(X)
            array([24,  0])

        """
        if fun_control is not None:
            self.fun_control = fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            # provoke error:
            if random() < 0.1:
                y = np.append(y, np.nan)
            else:
                y = np.append(y, np.sum(X[i]))
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            print(y)
            return y
