import numpy as np
from numpy.random import default_rng
from random import random


class analytical:
    """
    Analytical test functions.

    Args:
        offset (float): offset
        hz (float): hz
        seed (int): seed.
            See [Numpy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start)

    """

    def __init__(self, offset=0.0, hz=0, seed=126):
        self.offset = offset
        self.hz = hz
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.fun_control = {"sigma": 0, "seed": None, "sel_var": None}

    def add_noise(self, y):
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

    def fun_linear(self, X, fun_control=None):
        """Linear function.

        Args:
            X (array): input

        Returns:
            (float): objective function value.
        """
        if fun_control is not None:
            self.fun_control = fun_control
        try:
            X.shape[1]
        except ValueError as err:
            print("error message:", err)
            X = np.array(X)

        if len(X.shape) < 2:
            X = np.array([X])
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, np.sum(X[i]))
        if self.fun_control["sigma"] > 0:
            return self.add_noise(y)
        else:
            return y

    def fun_sphere(self, X, fun_control=None):
        """Sphere function.

        Args:
            X (array): input
            fun_control (dict): dict with entries `seed` and `sigma`.

        Returns:
            (float): function values
        """
        if fun_control is not None:
            self.fun_control = fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array(X)

        if len(X.shape) < 2:
            X = np.array([X])
        offset = np.ones(X.shape[1]) * self.offset
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, np.sum((X[i] - offset) ** 2))
        # TODO: move to a separate function:
        if self.fun_control["sigma"] > 0:
            # Use own rng:
            if self.fun_control["seed"] is not None:
                rng = default_rng(seed=fun_control["seed"])
            # Use class rng:
            else:
                rng = self.rng
            noise_y = np.array([], dtype=float)
            for y_i in y:
                noise_y = np.append(noise_y, y_i + rng.normal(loc=0, scale=fun_control["sigma"], size=1))
            return noise_y
        else:
            return y

    def fun_cubed(self, X, fun_control=None):
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array(X)

        if len(X.shape) < 2:
            X = np.array([X])
        offset = np.ones(X.shape[1]) * self.offset
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, np.sum((X[i] - offset) ** 3))
        # TODO: move to a separate function:
        if fun_control["sigma"] > 0:
            # Use own rng:
            if fun_control["seed"] is not None:
                rng = default_rng(seed=fun_control["seed"])
            # Use class rng:
            else:
                rng = self.rng
            noise_y = np.array([], dtype=float)
            for i in y:
                # noise_y = np.append(
                #     noise_y, i + np.random.normal(loc=0, scale=self.sigma, size=1)
                noise_y = np.append(noise_y, i + rng.normal(loc=0, scale=fun_control["sigma"], size=1))
            return noise_y
        else:
            return y

    def fun_forrester(self, X, fun_control=None):
        """
        Function used by [Forr08a, p.83].
        f(x) = (6x- 2)^2 sin(12x-4) for x in [0,1].
        Starts with three sample points at x=0, x=0.5, and x=1.

        Args:
            X (flooat): input values (1-dim)

        Returns:
            (float): function value
        """
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array(X)

        if len(X.shape) < 2:
            X = np.array([X])
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, (6.0 * X[i] - 2) ** 2 * np.sin(12 * X[i] - 4))
        # TODO: move to a separate function:
        if fun_control["sigma"] > 0:
            # Use own rng:
            if fun_control["seed"] is not None:
                rng = default_rng(seed=fun_control["seed"])
            # Use class rng:
            else:
                rng = self.rng
            noise_y = np.array([], dtype=float)
            for i in y:
                # noise_y = np.append(
                #     noise_y, i + np.random.normal(loc=0, scale=self.sigma, size=1)
                noise_y = np.append(noise_y, i + rng.normal(loc=0, scale=fun_control["sigma"], size=1))
            return noise_y
        else:
            return y

    def fun_branin(self, X, fun_control=None):
        """Branin function.

        The 2-dim Branin function is
        y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s,
        where values of a, b, c, r, s and t are: a = 1, b = 5.1 / (4*pi**2),
        c = 5 / pi, r = 6, s = 10 and t = 1 / (8*pi).

        It has three global minima:
        f(x) = 0.397887 at (-pi, 12.275), (pi, 2.275), and (9.42478, 2.475).

        Input Domain:
        This function is usually evaluated on the square  x1 in  [-5, 10] x x2 in [0, 15].

        Args:
            X (array): input value
            fun_control (dict): dict with entries `seed` and `sigma`.

        Returns:
            (float): function value

        """
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
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
        # TODO: move to a separate function:
        if fun_control["sigma"] > 0:
            # Use own rng:
            if fun_control["seed"] is not None:
                rng = default_rng(seed=fun_control["seed"])
            # Use class rng:
            else:
                rng = self.rng
            noise_y = np.array([], dtype=float)
            for i in y:
                # noise_y = np.append(
                #     noise_y, i + np.random.normal(loc=0, scale=self.sigma, size=1)
                noise_y = np.append(noise_y, i + rng.normal(loc=0, scale=fun_control["sigma"], size=1))
            return noise_y
        else:
            return y

    def fun_branin_factor(self, X, fun_control=None):
        """Branin function with factor variable x_3.

        The 2-dim Branin, or Branin-Hoo, function has three global minima.
        The recommended values of a, b, c, r, s and t are: a = 1, b = 5.1 / (4*pi**2),
        c = 5 / π, r = 6, s = 10 and t = 1 / (8*pi).

        Input Domain:
        This function is usually evaluated on the square  x1 in  [-5, 10] x x2 in [0, 15]
        and with x3 from the set {0, 1, 2}, i.e., x3 is a factor variable with three levels.

        Global Minimum:
        f(x) = 0.397887 -1  at (-pi, 12.275, 2), (pi, 2.275, 2), and (9.42478, 2.475, 2).

        Args:
            X (array): input value

        Returns:
            (float): function value

        """
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != 3:
            raise Exception
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
        # TODO: move to a separate function:
        if fun_control["sigma"] > 0:
            # Use own rng:
            if fun_control["seed"] is not None:
                rng = default_rng(seed=fun_control["seed"])
            # Use class rng:
            else:
                rng = self.rng
            noise_y = np.array([], dtype=float)
            for i in y:
                noise_y = np.append(noise_y, i + rng.normal(loc=0, scale=fun_control["sigma"], size=1))
            return noise_y
        else:
            return y

    def fun_branin_modified(self, X, fun_control=None):
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])

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
        # TODO: move to a separate function:
        if fun_control["sigma"] > 0:
            # Use own rng:
            if fun_control["seed"] is not None:
                rng = default_rng(seed=fun_control["seed"])
            # Use class rng:
            else:
                rng = self.rng
            noise_y = np.array([], dtype=float)
            for i in y:
                noise_y = np.append(noise_y, i + rng.normal(loc=0, scale=fun_control["sigma"], size=1))
            return noise_y
        else:
            return y

    def branin_noise(self, X):
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])

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
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != 2:
            raise Exception
        x0 = X[:, 0]
        x1 = X[:, 1]
        y = 2.0 * np.sin(x0 + self.hz) + 0.5 * np.cos(x1 + self.hz)
        # TODO: move to a separate function:
        if fun_control["sigma"] > 0:
            # Use own rng:
            if fun_control["seed"] is not None:
                rng = default_rng(seed=fun_control["seed"])
            # Use class rng:
            else:
                rng = self.rng
            noise_y = np.array([], dtype=float)
            for i in y:
                noise_y = np.append(noise_y, i + rng.normal(loc=0, scale=fun_control["sigma"], size=1))
            return noise_y
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

    #     if len(X.shape) < 2:
    #         X = np.array([X])
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

    def fun_runge(self, X, fun_control=None):
        """
        Runge function.
        Formula: f(x) = 1/ (1 + sum(x_i) - offset)^2
        Dim: k >= 1
        Interval: -5 <= x <= 5

        Args:
            X (numpy.array):
            input
            fun_control (dictionary, optional):
            control parameters. Defaults to None.

        Returns:
            (float) :
            function value
        """
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array(X)

        if len(X.shape) < 2:
            X = np.array([X])
        offset = np.ones(X.shape[1]) * self.offset
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, (1 / (1 + np.sum((X[i] - offset) ** 2))))
        # TODO: move to a separate function:
        if fun_control["sigma"] > 0:
            # Use own rng:
            if fun_control["seed"] is not None:
                rng = default_rng(seed=fun_control["seed"])
            # Use class rng:
            else:
                rng = self.rng
            noise_y = np.array([], dtype=float)
            for i in y:
                noise_y = np.append(noise_y, i + rng.normal(loc=0, scale=fun_control["sigma"], size=1))
            return noise_y
        else:
            return y

    def fun_wingwt(self, X, fun_control=None):
        """
        Wing weight function. Example from Forrester et al. to understand the weight
        of an unpainted light aircraft wing as a function of nine design and operational parameters:
        W = 0.036 S_W**0.758 * Wfw**0.0035 ( A / (cos**2 Lambda))**0.6 *
            q**0.006  * lambda**0.04 * ( (100 Rtc)/(cos Lambda) ))**-0.3*
            (Nz Wdg)**0.49

        | Symbol    | Parameter                              | Baseline | Minimum | Maximum |
        |-----------|----------------------------------------|----------|---------|---------|
        | $S_W$     | Wing area ($ft^2$)                     | 174      | 150     | 200     |
        | $W_{fw}$  | Weight of fuel in wing (lb)            | 252      | 220     | 300     |
        | $A$       | Aspect ratio                          | 7.52     | 6       | 10      |
        | $Lambda$ | Quarter-chord sweep (deg)              | 0        | -10     | 10      |
        | $q$       | Dynamic pressure at cruise ($lb/ft^2$) | 34       | 16      | 45      |
        | $lambda$ | Taper ratio                            | 0.672    | 0.5     | 1       |
        | $R_{tc}$  | Aerofoil thickness to chord ratio      | 0.12     | 0.08    | 0.18    |
        | $N_z$     | Ultimate load factor                   | 3.8      | 2.5     | 6       |
        | $W_{dg}$  | Flight design gross weight (lb)         | 2000     | 1700    | 2500    |
        | $W_p$     | paint weight (lb/ft^2)                   | 0.064 |   0.025  | 0.08    |

        Args:
            X (numpy.array):
                10-dim input vector
            fun_control (dictionary, optional):
                control parameters. Defaults to None.

        Returns:
            (float) :
            function value
        """
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array(X)
        #
        W_res = np.array([], dtype=float)
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
            W_res = np.append(W_res, W)
        return W_res

    def fun_xsin(self, X, fun_control=None):
        """
        Args:
            X (float): input values (1-dim)

        Returns:
            (float): function value
        """
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array(X)

        if len(X.shape) < 2:
            X = np.array([X])
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, X[i] * np.sin(1.0 / X[i]))
        # TODO: move to a separate function:
        if fun_control["sigma"] > 0:
            # Use own rng:
            if fun_control["seed"] is not None:
                rng = default_rng(seed=fun_control["seed"])
            # Use class rng:
            else:
                rng = self.rng
            noise_y = np.array([], dtype=float)
            for i in y:
                # noise_y = np.append(
                #     noise_y, i + np.random.normal(loc=0, scale=self.sigma, size=1)
                noise_y = np.append(noise_y, i + rng.normal(loc=0, scale=fun_control["sigma"], size=1))
            return noise_y
        else:
            return y

    def fun_rosen(self, X, fun_control=None):
        if fun_control is None:
            fun_control = self.fun_control
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
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

    def fun_random_error(self, X, fun_control=None):
        """Return errors for testing spot stability.

        Args:
            X (array): input

        Returns:
            (float): objective function value.
        """
        if fun_control is not None:
            self.fun_control = fun_control
        try:
            X.shape[1]
        except ValueError as err:
            print("error message:", err)
            X = np.array(X)
        if len(X.shape) < 2:
            X = np.array([X])
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
