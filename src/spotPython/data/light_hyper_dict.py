import json
from . import base


class LightHyperDict(base.FileConfig):
    """Lightning hyperparameter dictionary."""

    def __init__(self):
        super().__init__(
            filename="light_hyper_dict.json",
        )

    def load(self):
        with open(self.path, "r") as f:
            d = json.load(f)
        return d
