from .designs import designs
from scipy.stats.qmc import LatinHypercube
from numpy import repeat
from spotPython.utils.transform import scale
from numpy import zeros
from numpy import ones


class spacefilling(designs):
    def __init__(self, k=2, seed=123):
        """
        Spacefilling design class

        Args:
            k (int, optional): number of design variables (dimensions). Defaults to 2.
            seed (int, optional): random seed. Defaults to 123.
        """
        self.k = k
        self.seed = seed
        super().__init__(k, seed)
        self.sampler = LatinHypercube(d=self.k, seed=self.seed)

    def scipy_lhd(self, n, repeats=1, lower=None, upper=None):
        """
        Latin hypercube sampling based on scipy.

        Args:
            n (int): number of samples
            repeats (int): number of repeats (replicates)
            lower (int, optional): lower bound. Defaults to 0.
            upper (int, optional): upper bound. Defaults to 1.

        Returns:
            (numpy.ndarray): Latin hypercube design.
        """
        if lower is None:
            lower = zeros(self.k)
        if upper is None:
            upper = ones(self.k)
        sample = self.sampler.random(n=n)
        des = scale(sample, lower, upper)
        return repeat(des, repeats, axis=0)
