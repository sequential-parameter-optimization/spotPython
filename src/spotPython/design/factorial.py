import numpy as np
from numpy import mgrid
from .designs import designs


class factorial(designs):
    """
    Super class for factorial designs.

    Attributes:
        k (int): The number of factors.
        seed (int): The seed for the random number generator.
    """

    def __init__(self, k: int = 2, seed: int = 123) -> None:
        """
        Initializes a factorial design object.

        Args:
            k (int): The number of factors. Defaults to 2.
            seed (int): The seed for the random number generator. Defaults to 123.
        """
        super().__init__(k, seed)

    def full_factorial(self, p: int) -> "np.ndarray":
        """
        Generates a full factorial design.

        Args:
            p (int): The number of levels for each factor.

        Returns:
            numpy.ndarray: A 2D array representing the full factorial design.

        Examples:
            >>> from spotpython.design.factorial import factorial
                factorial_design = factorial(k=2)
                factorial_design.full_factorial(p=2)
                array([[0., 0.],
                    [0., 1.],
                    [1., 0.],
                    [1., 1.]])
        """
        i = (slice(0, 1, p * 1j),) * self.k
        return mgrid[i].reshape(self.k, p**self.k).T
