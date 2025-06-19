import numpy as np
from spotpython.design.designs import Designs
from scipy.stats import qmc


class Sobol(Designs):
    """
    Super class for sobol designs.

    Attributes:
        k (int): The number of factors.
        seed (int): The seed for the random number generator.
    """

    def __init__(self, k: int = 2, seed: int = 123) -> None:
        """
        Initializes a sobol design object.

        Args:
            k (int): The number of factors (dimension). Defaults to 2.
            seed (int): The seed for the random number generator. Defaults to 123.
        """
        super().__init__(k, seed)
        self.k = k
        self.seed = seed

    def generate_sobol_design(self, n_points: int, seed: int = None) -> np.ndarray:
        """Generates a Sobol sequence design

        Args:
            n_points (int):
                The number of points to generate in the Sobol sequence.
            seed (Optional[int]):
                The seed for the random number generator.
                If None, uses the instance's seed.

        Returns:
            np.ndarray: An array of shape (n_points, n_dim) containing the generated Sobol sequence points.

        Notes:
            - The Sobol sequence is generated with a length that is a power of 2. The function generates at least n_points and returns the first n_points.
            - For n_points not being a power of 2, extra points are generated and truncated.
            - Scrambling is enabled for improved uniformity.

        Examples:
            >>> from spotpython.design.sobol import Sobol
            >>> sobol_design = Sobol(k=3, seed=42)
            >>> sobol_points = sobol_design.generate_sobol_design(n_points=10)
            >>> print(sobol_points.shape)
            (10, 3)
        """
        if seed is not None:
            self.seed = seed
        sampler = qmc.Sobol(d=self.k, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(n_points)))
        return sampler.random_base2(m=m)[:n_points, :]
