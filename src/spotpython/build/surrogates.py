from numpy.random import default_rng


class surrogates:
    """
    Super class for all surrogate model classes (e.g., Kriging)
    """
    def __init__(self, name="", seed=123, verbosity=0):
        self.name = name
        self.seed = seed
        self.rng = default_rng(self.seed)
        self.log = {}
        self.verbosity = verbosity
