import json
from spotPython.data import base


class LightHyperDict(base.FileConfig):
    """Lightning hyperparameter dictionary.

    This class extends the FileConfig class to provide a dictionary for storing hyperparameters.

    Attributes:
        filename (str):
            The name of the file where the hyperparameters are stored.
    """

    def __init__(self):
        """Initialize the LightHyperDict object.

        Examples:
            >>> lhd = LightHyperDict()
        """
        super().__init__(
            filename="light_hyper_dict.json",
        )

    def load(self) -> dict:
        """Load the hyperparameters from the file.

        Returns:
            dict: A dictionary containing the hyperparameters.

        Examples:
            >>> lhd = LightHyperDict()
            >>> hyperparams = lhd.load()
            >>> print(hyperparams)
            {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10}
        """
        with open(self.path, "r") as f:
            d = json.load(f)
        return d
