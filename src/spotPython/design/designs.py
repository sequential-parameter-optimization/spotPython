from numpy.random import default_rng


class designs:
    """
    Super class for all design classes (factorial and spacefilling)
    """

    def __init__(self, k=2, seed=123):
        self.designs = []
        self.k = k
        self.seed = seed
        self.rng = default_rng(self.seed)

    def get_dim(self):
        """Return design dimension."""
        print(self.k)
