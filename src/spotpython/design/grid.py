import numpy as np
from spotpython.design.designs import Designs


class Grid(Designs):
    """
    Super class for grid designs.

    Attributes:
        k (int): The number of factors (dimension).
        seed (int): The seed for the random number generator.
    """

    def __init__(self, k: int = 2, seed: int = 123) -> None:
        """
        Initializes a grid design object.

        Args:
            k (int): The number of factors. Defaults to 2.
            seed (int): The seed for the random number generator. Defaults to 123.
        """
        super().__init__(k, seed)
        self.k = k
        self.seed = seed

    def generate_grid_design(self, points_per_dim: int) -> np.ndarray:
        """Generates a regular grid design.

        Args:
            points_per_dim (int): The number of points per dimension.

        Returns:
            numpy.ndarray: A 2D array of shape (points_per_dim^n_dim, n_dim) with grid points.

        Examples:
            >>> from spotpython.design.grid import Grid
            >>> grid_design = Grid(k=2)
            >>> grid_points = grid_design.generate_grid_design(points_per_dim=5)
            >>> print(grid_points)
            [[0.  0. ]
             [0.  0.25]
             [0.  0.5 ]
             ...
             [1.   1. ]]

        """
        if self.k != 2:
            raise ValueError("Grid design currently implemented for 2D only for simplicity.")
        ticks = np.linspace(0, 1, points_per_dim, endpoint=True)
        x, y = np.meshgrid(ticks, ticks)
        return np.vstack([x.ravel(), y.ravel()]).T
