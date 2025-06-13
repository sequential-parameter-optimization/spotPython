import numpy as np
from spotpython.design.designs import Designs


class Poor(Designs):
    """
    Super class for poorly projected (collinear) designs.

    Attributes:
        k (int): The number of factors.
        seed (int): The seed for the random number generator.
    """

    def __init__(self, k: int = 2, seed: int = 123) -> None:
        """
        Initializes a Poor design object.

        Args:
            k (int): The number of factors. Defaults to 2.
            seed (int): The seed for the random number generator. Defaults to 123.
        """
        super().__init__(k, seed)
        self.k = k
        self.seed = seed

    def generate_collinear_design(self, n_points: int) -> np.ndarray:
        """Generates a collinear design (poorly projected).

        Args:
            n_points (int): The number of points to generate.

        Returns:
            numpy.ndarray: A 2D array of shape (n_points, n_dim) with collinear points.

        Examples:
            >>> from spotpython.design.poor import Poor
            >>> poor_design = Poor(k=2)
            >>> collinear_points = poor_design.generate_collinear_design(n_points=10)
            >>> print(collinear_points)
            [[0.1  0.5 ]
             [0.2  0.5 ]
             [0.3  0.5 ]
             ...
             [0.9  0.5 ]]

        """
        if self.k != 2:
            raise ValueError("Collinear design currently implemented for 2D only.")
        x_coords = np.linspace(0.1, 0.9, n_points)
        y_coords = np.full_like(x_coords, 0.5)  # All points on y=0.5 line
        # y_coords = 0.2 * x_coords + 0.3  # Or a sloped line
        return np.vstack([x_coords, y_coords]).T
