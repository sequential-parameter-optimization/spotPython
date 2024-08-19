from numpy.random import default_rng
from typing import List


class designs:
    """
    Super class for all design classes (factorial and spacefilling).

    Attributes:
        designs (List):
            A list of designs.
        k (int):
            The dimension of the design.
        seed (int):
            The seed for the random number generator.
        rng (Generator):
            A random number generator instance.
    """

    def __init__(self, k: int = 2, seed: int = 123) -> None:
        """
        Initializes a Designs object with the given dimension and seed.

        Args:
            k (int):
                The dimension of the design. Defaults to 2.
            seed (int):
                The seed for the random number generator. Defaults to 123.
        Examples:
            >>> designs = designs(k=2, seed=123)
            >>> designs.get_dim()
            2

        """
        self.designs: List = []
        self.k: int = k
        self.seed: int = seed
        self.rng = default_rng(self.seed)

    def get_dim(self) -> int:
        """
        Returns the dimension of the design.

        Returns:
            int: The dimension of the design.
        """
        return self.k
