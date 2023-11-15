import json
from spotPython.data import base


class SklearnHyperDict(base.FileConfig):
    """Scikit-learn hyperparameter dictionary.

    This class extends the FileConfig class to provide a dictionary for storing hyperparameters.

    Attributes:
        filename (str): The name of the file where the hyperparameters are stored.
    """

    def __init__(self):
        """Initialize the SklearnHyperDict object.

        Examples:
            >>> shd = SklearnHyperDict()
        """
        super().__init__(
            filename="sklearn_hyper_dict.json",
        )

    def load(self) -> dict:
        """Load the hyperparameters from the file.

        Returns:
            (dict): A dictionary containing the hyperparameters.
        Examples:
            >>> shd = SklearnHyperDict()
            >>> hyperparams = shd.load()
            >>> print(hyperparams)
            {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10}
        """
        with open(self.path, "r") as f:
            d = json.load(f)
        return d


# Example usage
if __name__ == "__main__":
    # Create a SklearnHyperDict object
    shd = SklearnHyperDict()

    # Load the hyperparameters from the file
    hyperparams = shd.load()

    # Print the hyperparameters
    print(hyperparams)
