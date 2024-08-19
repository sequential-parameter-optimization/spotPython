import json
from spotpython.data import base
import pathlib


class SklearnHyperDict(base.FileConfig):
    """Scikit-learn hyperparameter dictionary.

    This class extends the FileConfig class to provide a dictionary for storing hyperparameters.

    Attributes:
        filename (str): The name of the file where the hyperparameters are stored.
    """

    def __init__(
        self,
        filename: str = "sklearn_hyper_dict.json",
        directory: None = None,
    ) -> None:
        super().__init__(filename=filename, directory=directory)
        self.filename = filename
        self.directory = directory
        self.hyper_dict = self.load()

    @property
    def path(self):
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

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
