import numpy as np
from spotpython.design.designs import Designs


class Random(Designs):
    """
    Super class for random designs.

    Attributes:
        k (int): The number of factors.
        seed (int): The seed for the random number generator.
    """

    def __init__(self, k: int = 2, seed: int = 123) -> None:
        """
        Initializes a random design object.

        Args:
            k (int): The number of factors. Defaults to 2.
            seed (int): The seed for the random number generator. Defaults to 123.
        """
        super().__init__(k, seed)
        self.k = k
        self.seed = seed

    def uniform(self, n_points: int, seed: int = None) -> np.ndarray:
        """
        Generates a random design using uniform distribution.

        Args:
            n_points (int): The number of points to generate.
            seed (int, optional): The seed for the random number generator. If None, uses the instance's seed.

        Returns:
            numpy.ndarray: A 2D array of shape (n_points, k) with random values in [0, 1).

        Examples:
            >>> from spotpython.design.random import Random
            >>> random_design = Random(k=3)
            >>> random_design.uniform(n_points=5)
            array([[0.123, 0.456, 0.789],
                   [0.234, 0.567, 0.890],
                   [0.345, 0.678, 0.901],
                   [0.456, 0.789, 0.012],
                   [0.567, 0.890, 0.123]])

        """
        if seed is not None:
            seed = self.seed
        rng = np.random.default_rng(seed)
        return rng.random((n_points, self.k))
