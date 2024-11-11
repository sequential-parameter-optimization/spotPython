from numpy.random import default_rng
from typing import List


class Designs:
    """
    Super class for all design classes (factorial and space-filling).

    Attributes:
        designs (List): A list of design instances.
        k (int): The dimension of the design.
        seed (int): The seed for the random number generator.
        rng (Generator): A random number generator instance.
    """

    def __init__(self, k: int, seed: int = 123) -> None:
        """
        Initializes a Designs object with the given dimension and seed.

        Args:
            k (int): The dimension of the design.
            seed (int): The seed for the random number generator. Defaults to 123.

        Raises:
            ValueError: If 'k' is not an integer.

        Examples:
            >>> from spotpython.design.designs import Designs
            >>> designs = Designs(k=2, seed=123)
            >>> designs.get_dim()
            2
        """
        if not isinstance(k, int):
            raise ValueError("The dimension of the design must be an integer.")

        self.k: int = k
        self.seed: int = seed
        self.rng = default_rng(self.seed)
        self.designs: List = []

    def get_dim(self) -> int:
        """
        Returns the dimension of the design.

        Returns:
            int: The dimension of the design.
        """
        return self.k
