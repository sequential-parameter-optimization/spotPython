import numpy as np
from scipy.stats.qmc import LatinHypercube
from spotpython.utils.transform import scale
from typing import Optional, Union
from spotpython.design.designs import Designs
from scipy.stats import qmc


class SpaceFilling(Designs):
    """
    A class for generating space-filling designs using Latin Hypercube Sampling.
    """

    def __init__(
        self,
        k: int,
        scramble: bool = True,
        strength: int = 1,
        optimization: Optional[Union[None, str]] = None,
        seed: int = 123,
    ) -> None:
        """
        Initializes a SpaceFilling design class.
        Based on scipy.stats.qmc's LatinHypercube method.

        Args:
            k (int):
                Dimension of the parameter space.
            scramble (bool, optional):
                When False, center samples within cells of a multi-dimensional grid.
                Otherwise, samples are randomly placed within cells of the grid.
                Note:
                    Setting `scramble=False` does not ensure deterministic output. For that, use the `seed` parameter.
                Default is True.
            optimization (Optional[Union[None, str]]):
                Whether to use an optimization scheme to improve the quality after sampling.
                Note that this is a post-processing step that does not guarantee that all
                properties of the sample will be conserved.
                Defaults to None.
                Options:
                    - "random-cd": Random permutations of coordinates to lower the centered discrepancy. The best sample based on the centered discrepancy is constantly updated.
                    Centered discrepancy-based sampling shows better space-filling
                    robustness toward 2D and 3D subprojections compared to using other discrepancy measures.
                    - "lloyd": Perturb samples using a modified Lloyd-Max algorithm. The process converges to equally spaced samples.
            strength (Optional[int]):
                Strength of the LHS. `strength=1` produces a plain LHS while `strength=2` produces an orthogonal array based LHS of strength 2.
                In that case, only `n=p**2` points can be sampled, with `p` a prime number.
                It also constrains `d <= p + 1`.
                Defaults to 1.
            seed (int, optional):
                Seed for the random number generator. Defaults to 123.
        """
        super().__init__(k=k, seed=seed)
        self.sampler = LatinHypercube(d=self.k, scramble=scramble, strength=strength, optimization=optimization, seed=seed)

    def scipy_lhd(
        self,
        n: int,
        repeats: int = 1,
        lower: Optional[Union[int, float, np.ndarray]] = None,
        upper: Optional[Union[int, float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Latin hypercube sampling based on scipy.

        Args:
            n (int): Number of samples.
            repeats (int): Number of repeats (replicates).
            lower (int, float, or np.ndarray, optional): Lower bound. Defaults to a zero vector.
            upper (int, float, or np.ndarray, optional): Upper bound. Defaults to a one vector.

        Returns:
            np.ndarray: Latin hypercube design with specified dimensions and boundaries.

        Examples:
            >>> from spotpython.design.spacefilling import SpaceFilling
            >>> lhd = SpaceFilling(k=2, seed=123)
            >>> lhd.scipy_lhd(n=5, repeats=2, lower=np.array([0, 0]), upper=np.array([1, 1]))
            array([[0.66352963, 0.5892358 ],
                   [0.66352963, 0.5892358 ],
                   [0.55592803, 0.96312564],
                   [0.55592803, 0.96312564],
                   [0.16481882, 0.0375811 ],
                   [0.16481882, 0.0375811 ],
                   [0.215331  , 0.34468512],
                   [0.215331  , 0.34468512],
                   [0.83604909, 0.62202146],
                   [0.83604909, 0.62202146]])
        """
        if lower is None:
            lower = np.zeros(self.k)
        if upper is None:
            upper = np.ones(self.k)

        sample = self.sampler.random(n=n)
        des = scale(sample, lower, upper)
        return np.repeat(des, repeats, axis=0)

    def generate_qms_lhs_design(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Generates a Latin Hypercube Sampling design using the `scipy.stats.qmc` module.
        Generates a Latin Hypercube Sampling (LHS) design with the specified number of points
        and dimensions.

        Args:
            n_points (int): The number of points to generate.
            seed (Optional[int]):
                Seed for the random number generator to ensure reproducibility.
                Defaults to None. If None, uses the seed specified during initialization.

        Returns:
            np.ndarray: An array of shape (n_points, n_dim) containing the generated Latin Hypercube Sampling points.

        Notes:
            - The Latin Hypercube Sampling is generated with a specified number of points and dimensions.
            - The points are uniformly distributed across the unit hypercube [0, 1]^n_dim.

        Examples:
            >>> from spotpython.design.spacefilling import SpaceFilling
            >>> lhs_design = SpaceFilling(k=3, seed=42)
            >>> lhs_points = lhs_design.generate_qms_lhs_design(n_points=10)
            >>> print(lhs_points.shape)
            (10, 3)
        """
        if seed is None:
            seed = self.seed
        sampler = qmc.LatinHypercube(d=self.k, seed=seed)
        return sampler.random(n=n_points)
