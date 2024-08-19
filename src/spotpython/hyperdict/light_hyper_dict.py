import json
from spotpython.data import base
import pathlib


class LightHyperDict(base.FileConfig):
    """Lightning hyperparameter dictionary.

    This class extends the FileConfig class to provide a dictionary for storing hyperparameters.

    Attributes:
        filename (str):
            The name of the file where the hyperparameters are stored.
    """

    def __init__(
        self,
        filename: str = "light_hyper_dict.json",
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
            dict: A dictionary containing the hyperparameters.

        Examples:
            >>> from spotpython.hyperdict.light_hyper_dict import LightHyperDict
                lhd = LightHyperDict()
                lhd.hyper_dict
                {'NetLightRegression': {'l1': {'type': 'int',
                'default': 3,
                'transform': 'transform_power_2_int',
                'lower': 3,
                'upper': 8},
                'epochs': {'type': 'int',
                'default': 4,
                'transform': 'transform_power_2_int',
                'lower': 4,
                'upper': 9},
                ...
                'transform': 'None',
                'class_name': 'torch.optim',
                'core_model_parameter_type': 'str',
                'lower': 0,
                'upper': 11}}}
            # Assume the user specified file `user_hyper_dict.json` is in the `./hyperdict/` directory.
            >>> user_lhd = LightHyperDict(filename='user_hyper_dict.json', directory='./hyperdict/')
        """
        with open(self.path, "r") as f:
            d = json.load(f)
        return d
