import numpy as np
from numpy.random import default_rng
from typing import List, Optional, Dict


class Analytical:
    """
    Class for analytical test functions.

    Args:
        offset (float):
            Offset value. Defaults to 0.0.
        seed (int):
            Seed value for random number generation. Defaults to 126.
        fun_control (dict):
            Dictionary containing control parameters for the function. Defaults to None.

    Notes:
        See [Numpy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start)

    Attributes:
        offset (float):
            Offset value.
        sigma (float):
            Noise level.
        seed (int):
            Seed value for random number generation.
        rng (Generator):
            Numpy random number generator object.
        fun_control (dict):
            Dictionary containing control parameters for the function.
    """

    def __init__(self, offset: float = 0.0, sigma=0.0, seed: int = 126, fun_control=None) -> None:
        self.offset = offset
        self.sigma = sigma
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.fun_control = {"offset": offset, "sigma": self.sigma, "seed": self.seed}
        # overwrite fun_control with user input if provided
        if fun_control is not None:
            self.fun_control = fun_control
        # check if fun_control contains offset, sigma and seed, if not, add them
        if "offset" not in self.fun_control:
            self.fun_control["offset"] = self.offset
        if "sigma" not in self.fun_control:
            self.fun_control["sigma"] = self.sigma
        if "seed" not in self.fun_control:
            self.fun_control["seed"] = self.seed

    def __repr__(self) -> str:
        return f"analytical(offset={self.offset}, sigma={self.sigma}, seed={self.seed})"

    def _prepare_input_data(self, X, fun_control):
        if fun_control is not None:
            self.fun_control = fun_control
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = np.atleast_2d(X)
        return X

    def _add_noise(self, y: List[float]) -> np.ndarray:
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
                fun._add_noise(y)
            array([0.01087865, 1.63221335, 4.28792526, 4.19397442, 5.9202309 ])

        """
        if self.fun_control["sigma"] > 0:
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
        else:
            return y

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
        X = self._prepare_input_data(X, fun_control)
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
        return self._add_noise(y)

    def fun_linear(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Linear function.

        Args:
            X (array):
                input
            fun_control (dict):
                dict with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 1D numpy array with shape (n,) containing the calculated values, which were obtained by
            summing the weighted input values after subtracting the offset. Noise can be added to the output. An intercept
            can be provided by setting the `alpha` key in the `fun_control` dictionary. If the `beta` key is provided, the
            weighted sum is computed. If `beta` is not provided, the sum of the input values is computed.

        Examples:
            >>> from spotpython.fun.objectivefunctions import Analytical
            >>> import numpy as np
            >>> # Without offset and without noise
            >>> user_fun = UserAnalytical()
            >>> X = np.array([[0, 0, 0], [1, 1, 1]])
            >>> results = user_fun.fun_user_function(X)
            >>> print(results)
            >>>
            >>> # With offset and without noise
            >>> user_fun = UserAnalytical(offset=1.0)
            >>> X = np.array([[0, 0, 0], [1, 1, 1]])
            >>> results = user_fun.fun_user_function(X)
            >>> print(results)
            >>>
            >>> # With offset and noise
            >>> user_fun = UserAnalytical(offset=1.0, sigma=0.1, seed=1)
            >>> X = np.array([[0, 0, 0], [1, 1, 1]])
            >>> results = user_fun.fun_user_function(X)
            >>> print(results)
            >>>
            >>> # Provide alpha (intercept), no beta
            >>> fun_control = {"alpha": 10.0}
            >>> fun.fun_linear(X, fun_control=fun_control)
            >>> array([16., 25.])
            >>>
            >>> # Provide alpha and beta (weighted sum with intercept)
            >>> fun_control = {"alpha": 2.0, "beta": [1.0, 2.0, 3.0]}
            >>> fun.fun_linear(X, fun_control=fun_control)
            array([14., 32.])
                [0. 3.]
                [3. 0.]
                [3.03455842 0.08216181]

        """
        X = self._prepare_input_data(X, fun_control)
        offset = np.ones(X.shape[1]) * self.offset

        alpha = self.fun_control.get("alpha", 0.0)
        beta = self.fun_control.get("beta", None)
        if beta is not None:
            # check if beta is a numpy array
            if not isinstance(beta, np.ndarray):
                # convert beta to numpy array of shape (n,), where n is the number of columns in X
                beta = np.array(beta)
            if beta.shape[0] != X.shape[1]:
                raise Exception("beta must have the same number of elements as the number of columns in X")

        # Compute the linear response
        if beta is not None:
            # Weighted sum with intercept
            y = alpha + np.dot(X - offset, beta)
        else:
            # Original behavior: just sum the rows
            y = alpha + np.sum(X - offset, axis=1)

        return self._add_noise(y)

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
        X = self._prepare_input_data(X, fun_control)
        offset = np.ones(X.shape[1]) * self.offset
        y = np.sum((X - offset) ** 2, axis=1)
        return self._add_noise(y)

    def fun_cubed(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Cubed function. Implements the function f(x) = sum((x_i - offset)^3).

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
            >>> X = np.array([[1, 2, 3], [4, 5, 6], [-1, -1, -1]])
            >>> fun = analytical()
            >>> fun.fun_cubed(X)
            array([ 36., 405., -3.])
        """
        X = self._prepare_input_data(X, fun_control)
        offset = np.ones(X.shape[1]) * self.offset
        y = np.sum((X - offset) ** 3, axis=1)
        return self._add_noise(y)

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
        X = self._prepare_input_data(X, fun_control)
        y = ((6.0 * X - 2) ** 2) * np.sin(12 * X - 4)
        return self._add_noise(y)

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
        X = self._prepare_input_data(X, fun_control)
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
        return self._add_noise(y)

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
        X = self._prepare_input_data(X, fun_control)
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
        return self._add_noise(y)

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
        X = self._prepare_input_data(X, fun_control)
        if X.shape[1] != 2:
            raise Exception
        x0 = X[:, 0]
        x1 = X[:, 1]
        y = 2.0 * np.sin(x0 - self.offset) + 0.5 * np.cos(x1 - self.offset)
        return self._add_noise(y)

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
        X = self._prepare_input_data(X, fun_control)
        offset = np.ones(X.shape[1]) * self.offset
        squared_diff = (X - offset) ** 2
        sum_squared_diff = np.sum(squared_diff, axis=1)
        y = 1 / (1 + sum_squared_diff)
        return self._add_noise(y)

    def fun_wingwt_to_nat(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        r"""Wing weight function.
        Converts coded values to natural values, before applying the original `fun_wingwt` function (Eq. 1.4 in [Forr08a]).
        Calculate the weight of an unpainted light aircraft wing based on design and operational parameters.
        This function implements the wing weight model from Forrester et al., which aims to predict
        the wing weight \( W \) using the following formula:

        \[
        W = 0.036 \times S_W^{0.758} \times W_{fw}^{0.0035} \times \left( \frac{A}{\cos^2 \Lambda} \right)^{0.6}
        \times q^{0.006} \times \lambda^{0.04} \times \left( \frac{100 \times R_{tc}}{\cos \Lambda} \right)^{-0.3}
        \times (N_z \times W_{dg})^{0.49} + S_W \times W_p
        \]

        where:

        - \( S_W \): Wing area \((\text{ft}^2)\)
        - \( W_{fw} \): Weight of fuel in the wing (lb)
        - \( A \): Aspect ratio
        - \( \Lambda \): Quarter-chord sweep (degrees)
        - \( q \): Dynamic pressure at cruise \((\text{lb/ft}^2)\)
        - \( \lambda \): Taper ratio
        - \( R_{tc} \): Aerofoil thickness to chord ratio
        - \( N_z \): Ultimate load factor
        - \( W_{dg} \): Flight design gross weight (lb)
        - \( W_p \): Paint weight \((\text{lb/ft}^2)\)

        Parameter Overview:

        | Symbol    | Parameter                              | Baseline | Minimum | Maximum |
        |-----------|----------------------------------------|----------|---------|---------|
        | \( S_W \)     | Wing area \((\text{ft}^2)\)                     | 174      | 150     | 200     |
        | \( W_{fw} \)  | Weight of fuel in wing (lb)            | 252      | 220     | 300     |
        | \( A \)       | Aspect ratio                          | 7.52     | 6       | 10      |
        | \( \Lambda \) | Quarter-chord sweep (deg)              | 0        | -10     | 10      |
        | \( q \)       | Dynamic pressure at cruise \((\text{lb/ft}^2)\) | 34       | 16      | 45      |
        | \( \lambda \) | Taper ratio                            | 0.672    | 0.5     | 1       |
        | \( R_{tc} \)  | Aerofoil thickness to chord ratio      | 0.12     | 0.08    | 0.18    |
        | \( N_z \)     | Ultimate load factor                   | 3.8      | 2.5     | 6       |
        | \( W_{dg} \)  | Flight design gross weight (lb)        | 2000     | 1700    | 2500    |
        | \( W_p \)     | Paint weight \((\text{lb/ft}^2)\)      | 0.064 |   0.025  | 0.08    |

        Args:
            X (np.ndarray):
                A 2D numpy array where each row contains 10 parameters for which the wing weight will be calculated.
            fun_control (Optional[Dict]):
                A dictionary with keys `sigma` (noise level) and `seed` (random seed)
                for incorporating randomness if required. Default is `None`.

        Returns:
            np.ndarray:
            A 1D numpy array with shape (n,) containing the calculated wing weight values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([np.zeros(10), np.ones(10)])
            >>> fun = analytical()
            >>> fun.fun_wingwt(X)
            array([158.28245046, 409.33182691])
        """
        X = self._prepare_input_data(X, fun_control)
        Sw = X[:, 0] * 50 + 150  # equivalent to (200 - 150) + 150
        Wfw = X[:, 1] * 80 + 220  # equivalent to (300 - 220) + 220
        A = X[:, 2] * 4 + 6  # equivalent to (10 - 6) + 6
        L = (X[:, 3] * 20 - 10) * np.pi / 180  # equivalent to (10 - (-10)) - 10
        q = X[:, 4] * 29 + 16  # equivalent to (45 - 16) + 16
        la = X[:, 5] * 0.5 + 0.5  # equivalent to (1 - 0.5) + 0.5
        Rtc = X[:, 6] * 0.1 + 0.08  # equivalent to (0.18 - 0.08) + 0.08
        Nz = X[:, 7] * 3.5 + 2.5  # equivalent to (6 - 2.5) + 2.5
        Wdg = X[:, 8] * 800 + 1700  # equivalent to (2500 - 1700) + 1700
        Wp = X[:, 9] * 0.055 + 0.025  # equivalent to (0.08 - 0.025) + 0.025
        # Calculate W for all rows in a vectorized manner
        W = 0.036 * Sw**0.758 * Wfw**0.0035
        W *= (A / np.cos(L) ** 2) ** 0.6 * q**0.006
        W *= la**0.04
        print(f"W: {W}")
        print(f"(100 * Rtc / np.cos(L)): {(100 * Rtc / np.cos(L))}")
        W *= (100 * Rtc / np.cos(L)) ** (-0.3)
        print(f"W: {W}")
        W *= (Nz * Wdg) ** (0.49)
        W += Sw * Wp
        return self._add_noise(y=W)

    def fun_wingwt(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        r"""Wing weight function. Returns coded, not natural values.
        Calculate the weight of an unpainted light aircraft wing based on design and operational parameters.
        This function implements the wing weight model from Forrester et al., which aims to predict
        the wing weight \( W \) using the following formula:

        \[
        W = 0.036 \times S_W^{0.758} \times W_{fw}^{0.0035} \times \left( \frac{A}{\cos^2 \Lambda} \right)^{0.6}
        \times q^{0.006} \times \lambda^{0.04} \times \left( \frac{100 \times R_{tc}}{\cos \Lambda} \right)^{-0.3}
        \times (N_z \times W_{dg})^{0.49} + S_W \times W_p
        \]

        where:

        - \( S_W \): Wing area \((\text{ft}^2)\)
        - \( W_{fw} \): Weight of fuel in the wing (lb)
        - \( A \): Aspect ratio
        - \( \Lambda \): Quarter-chord sweep (degrees)
        - \( q \): Dynamic pressure at cruise \((\text{lb/ft}^2)\)
        - \( \lambda \): Taper ratio
        - \( R_{tc} \): Aerofoil thickness to chord ratio
        - \( N_z \): Ultimate load factor
        - \( W_{dg} \): Flight design gross weight (lb)
        - \( W_p \): Paint weight \((\text{lb/ft}^2)\)

        Parameter Overview:

        | Symbol    | Parameter                              | Baseline | Minimum | Maximum |
        |-----------|----------------------------------------|----------|---------|---------|
        | \( S_W \)     | Wing area \((\text{ft}^2)\)            | 174      | 150     | 200     |
        | \( W_{fw} \)  | Weight of fuel in wing (lb)            | 252      | 220     | 300     |
        | \( A \)       | Aspect ratio                          | 7.52     | 6       | 10      |
        | \( \Lambda \) | Quarter-chord sweep (deg)              | 0        | -10     | 10      |
        | \( q \)       | Dynamic pressure at cruise \((\text{lb/ft}^2)\) | 34       | 16      | 45      |
        | \( \lambda \) | Taper ratio                            | 0.672    | 0.5     | 1       |
        | \( R_{tc} \)  | Aerofoil thickness to chord ratio      | 0.12     | 0.08    | 0.18    |
        | \( N_z \)     | Ultimate load factor                   | 3.8      | 2.5     | 6       |
        | \( W_{dg} \)  | Flight design gross weight (lb)        | 2000     | 1700    | 2500    |
        | \( W_p \)     | Paint weight \((\text{lb/ft}^2)\)      | 0.064 |   0.025  | 0.08    |

        Args:
            X (np.ndarray):
                A 2D numpy array where each row contains 10 parameters for which the wing weight will be calculated.
            fun_control (Optional[Dict]):
                A dictionary with keys `sigma` (noise level) and `seed` (random seed)
                for incorporating randomness if required. Default is `None`.

        Returns:
            np.ndarray:
            A 1D numpy array with shape (n,) containing the calculated wing weight values.

        Examples:
            >>> from spotpython.fun.objectivefunctions import analytical
            >>> import numpy as np
            >>> X = np.array([np.zeros(10), np.ones(10)])
            >>> fun = analytical()
            >>> fun.fun_wingwt(X)
            array([158.28245046, 409.33182691])
        """
        X = self._prepare_input_data(X, fun_control)
        Sw = X[:, 0]
        Wfw = X[:, 1]
        A = X[:, 2]
        L = X[:, 3] * np.pi / 180
        q = X[:, 4]
        la = X[:, 5]
        Rtc = X[:, 6]
        Nz = X[:, 7]
        Wdg = X[:, 8]
        Wp = X[:, 9]
        # Calculate W for all rows in a vectorized manner
        W = 0.036 * Sw**0.758 * Wfw**0.0035
        W *= (A / np.cos(L) ** 2) ** 0.6 * q**0.006
        W *= la**0.04
        W *= (100 * Rtc / np.cos(L)) ** (-0.3)
        W *= (Nz * Wdg) ** (0.49)
        W += Sw * Wp
        return self._add_noise(y=W)

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
        X = self._prepare_input_data(X, fun_control)
        y = X * np.sin(1.0 / X)
        return self._add_noise(y)

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
        X = self._prepare_input_data(X, fun_control)
        if X.shape[1] != 2:
            raise Exception
        x0 = X[:, 0]
        x1 = X[:, 1]
        b = 10
        y = (x0 - 1) ** 2 + b * (x1 - x0**2) ** 2
        return self._add_noise(y)

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
        X = self._prepare_input_data(X, fun_control)
        # Compute the sum of rows of X
        y = np.sum(X, axis=1)
        # Determine which elements to set to np.nan
        nan_mask = self.rng.random(size=y.shape) < 0.1
        y[nan_mask] = np.nan

        return self._add_noise(y)
