import json
from spotPython.data import base


class TorchHyperDict(base.FileConfig):
    """PyTorch hyperparameter dictionary.

    This class extends the FileConfig class to provide a dictionary for storing hyperparameters.

    Attributes:
        filename (str): The name of the file where the hyperparameters are stored.
    """

    def __init__(self):
        """Initialize the TorchHyperDict object.
        Examples:
            >>> thd = TorchHyperDict()
        """
        super().__init__(
            filename="torch_hyper_dict.json",
        )

    def load(self) -> dict:
        """Load the hyperparameters from the file.

        Returns:
            (dict): A dictionary containing the hyperparameters.
        Examples:
            >>> thd = TorchHyperDict()
            >>> hyperparams = thd.load()
            >>> print(hyperparams)
            {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10}
        """
        with open(self.path, "r") as f:
            d = json.load(f)
        return d


# Example usage
if __name__ == "__main__":
    # Create a TorchHyperDict object
    thd = TorchHyperDict()

    # Load the hyperparameters from the file
    hyperparams = thd.load()

    # Print the hyperparameters
    print(hyperparams)
