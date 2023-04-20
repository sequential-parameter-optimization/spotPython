import json
from . import base


class SklearnHyperDict(base.FileConfig):
    """River hyperparameter dictionary."""

    def __init__(self):
        super().__init__(
            filename="sklearn_hyper_dict.json",
        )

    def load(self):
        with open(self.path, "r") as f:
            d = json.load(f)
        return d
