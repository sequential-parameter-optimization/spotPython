import numpy as np
from numpy.random import default_rng
from typing import List, Optional, Dict


class MultiAnalytical:
    """
    Class for multiobjective analytical test functions.

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
        m (int):
            Number of objectives.
    """

    def __init__(self, offset: float = 0.0, sigma=0.0, seed: int = 126, fun_control=None, m=1) -> None:
        self.offset = offset
        self.sigma = sigma
        self.m = m
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

    def fun_mo_linear(self, X: np.ndarray, fun_control: Optional[Dict] = None) -> np.ndarray:
        """Linear function with multi-objective support.

        Args:
            X (np.ndarray): Input array of shape (n, k), where n is the number of samples and k is the number of features.
            fun_control (dict): Dictionary with entries `sigma` (noise level) and `seed` (random seed).

        Returns:
            np.ndarray: A 2D numpy array with shape (n, m), where n is the number of samples and m is the number of objectives.

        Examples:
        >>> from spotpython.fun.multiobjectivefunctions import MultiAnalytical
            import numpy as np
            fun = MultiAnalytical(m=1)
            # Input data
            X = np.array([[0, 0, 0], [1, 1, 1]])
            # Single objective
            print(fun.fun_mo_linear(X))
            # Output: [[0.]
            #          [3.]]
            # Two objectives
            fun = MultiAnalytical(m=2)
            print(fun.fun_mo_linear(X))
            # Output: [[ 0. -0.]
            #          [ 3. -3.]]
            # Three objectives
            fun = MultiAnalytical(m=3)
            print(fun.fun_mo_linear(X))
            # Output: [[ 0. -0.  0.]
            #          [ 3. -3.  3.]]
            # Four objectives
            fun = MultiAnalytical(m=4)
            print(fun.fun_mo_linear(X))
            # Output: [[ 0. -0.  0. -0.]
            #          [ 3. -3.  3. -3.]]
        """
        X = self._prepare_input_data(X, fun_control)
        offset = np.ones(X.shape[1]) * self.offset

        alpha = self.fun_control.get("alpha", 0.0)
        beta = self.fun_control.get("beta", None)
        if beta is not None:
            # Check if beta is a numpy array
            if not isinstance(beta, np.ndarray):
                # Convert beta to numpy array of shape (n,), where n is the number of columns in X
                beta = np.array(beta)
            if beta.shape[0] != X.shape[1]:
                raise Exception("beta must have the same number of elements as the number of columns in X")

        # Compute the linear response
        if beta is not None:
            # Weighted sum with intercept
            y_0 = alpha + np.dot(X - offset, beta)
        else:
            # Original behavior: just sum the rows
            y_0 = alpha + np.sum(X - offset, axis=1)

        # Add noise to the primary objective
        y_0 = self._add_noise(y_0)

        # Generate multi-objective outputs
        objectives = [y_0 if i % 2 == 0 else -y_0 for i in range(self.m)]
        return np.column_stack(objectives)
