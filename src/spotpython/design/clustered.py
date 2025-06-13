import numpy as np
from spotpython.design.designs import Designs
from typing import Optional
from sklearn.datasets import make_blobs


class Clustered(Designs):
    """
    Super class for clustered designs.

    Attributes:
        k (int): The number of factors.
        seed (int): The seed for the random number generator.
    """

    def __init__(self, k: int = 2, seed: int = 123) -> None:
        """
        Initializes a clustered design object.

        Args:
            k (int): The number of factors. Defaults to 2.
            seed (int): The seed for the random number generator. Defaults to 123.
        """
        super().__init__(k, seed)
        self.k = k
        self.seed = seed

    def generate_clustered_design(self, n_points: int, n_clusters: int, seed: Optional[int] = None) -> np.ndarray:
        """Generates a clustered design.

        Args:
            n_points (int): The number of points to generate.
            n_clusters (int): The number of clusters.
            seed (Optional[int]): Optional seed for reproducibility.

        Returns:
            numpy.ndarray: A 2D array of shape (n_points, n_dim) with clustered points.

        Examples:
            >>> from spotpython.design.clustered import Clustered
            >>> clustered_design = Clustered(k=3)
            >>> clustered_design.generate_clustered_design(n_points=100, n_clusters=5, seed=42)
            array([[0.12, 0.34, 0.56],
                   [0.23, 0.45, 0.67],
                   ...])
        """
        X, _ = make_blobs(n_samples=n_points, n_features=self.k, centers=n_clusters, cluster_std=0.05, random_state=seed, center_box=(0.1, 0.9))
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        if np.any(X_min < 0) or np.any(X_max > 1):
            X = (X - X_min) / (X_max - X_min + 1e-6)
        return X
