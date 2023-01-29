from numpy import mgrid
from .designs import designs


class factorial(designs):
    """
    Super class for factorial designs.
    """

    def __init__(self, k=2, seed=123):
        super().__init__(k, seed)

    def full_factorial(self, p):
        i = (slice(0, 1, p * 1j),) * self.k
        return mgrid[i].reshape(self.k, p**self.k).T
