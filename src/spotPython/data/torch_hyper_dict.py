import json
from . import base


class TorchHyperDict(base.FileConfig):
    """Torch hyperparameter dictionary."""

    def __init__(self):
        super().__init__(
            filename="torch_hyper_dict.json",
        )

    def load(self):
        with open(self.path, "r") as f:
            d = json.load(f)
        return d
