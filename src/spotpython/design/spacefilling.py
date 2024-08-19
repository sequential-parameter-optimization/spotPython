from numpy import ones, repeat, zeros, ndarray
from scipy.stats.qmc import LatinHypercube
from spotpython.utils.transform import scale
from typing import Optional, Union

from .designs import designs


class spacefilling(designs):
    def __init__(self, k: int = 2, seed: int = 123) -> None:
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

    def scipy_lhd(
        self,
        n: int,
        repeats: int = 1,
        lower: Optional[Union[int, float]] = None,
        upper: Optional[Union[int, float]] = None,
    ) -> ndarray:
        """
        Latin hypercube sampling based on scipy.

        Args:
            n (int): number of samples
            repeats (int): number of repeats (replicates)
            lower (int or float, optional): lower bound. Defaults to 0.
            upper (int or float, optional): upper bound. Defaults to 1.

        Returns:
            (ndarray): Latin hypercube design.

        Examples:
            >>> from spotpython.design.spacefilling import spacefilling
                import numpy as np
                lhd = spacefilling(k=2, seed=123)
                lhd.scipy_lhd(n=5, repeats=2, lower=np.array([0,0]), upper=np.array([1,1]))
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
            lower = zeros(self.k)
        if upper is None:
            upper = ones(self.k)
        sample = self.sampler.random(n=n)
        des = scale(sample, lower, upper)
        return repeat(des, repeats, axis=0)
