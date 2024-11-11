import abc
import inspect
import itertools
import pathlib
import re
import shutil
import tarfile
import typing
import zipfile
from urllib import request
from pathlib import Path
from os import environ, path
from typing import Optional, Union


__all__ = ["Dataset", "SyntheticDataset", "FileDataset", "RemoteDataset"]

REG = "Regression"
BINARY_CLF = "Binary classification"
MULTI_CLF = "Multi-class classification"
MO_BINARY_CLF = "Multi-output binary classification"
MO_REG = "Multi-output regression"


def get_data_home(data_home: Optional[Union[str, Path]] = None) -> Path:
    """Return the location where remote datasets are to be stored.

    By default the data directory is set to a folder named 'spotriver_data' in the
    user home folder. Alternatively, it can be set by the 'SPOTRIVER_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.

    Args:
        data_home (str or pathlib.Path, optional):
            The path to spotriver data directory. If `None`, the default path
            is `~/spotriver_data`.

    Returns:
        data_home (pathlib.Path):
            The path to the spotriver data directory.

    Examples:
        >>> from pathlib import Path
        >>> get_data_home()
        PosixPath('/home/user/spotriver_data')
        >>> get_data_home(Path('/tmp/spotriver_data'))
        PosixPath('/tmp/spotriver_data')
    """
    if data_home is None:
        data_home = environ.get("SPOTRIVER_DATA", Path.home() / "spotriver_data")
    # Ensure data_home is a Path() object pointing to an absolute path
    data_home = Path(data_home).absolute()
    # Create data directory if it does not exists.
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


class Config(abc.ABC):
    """Base class for all configurations.

    All configurations inherit from this class, be they stored in a file or generated on the fly.

    Attributes:
        desc (str): The description from the docstring.
        _repr_content (dict): The items that are displayed in the __repr__ method.
    """

    def __init__(self):
        """Initialize a Config object."""
        pass

    @property
    def desc(self) -> str:
        """Return the description from the docstring.

        Returns:
            str: The description from the docstring.

        Examples:
            >>> class MyConfig(Config):
            ...     '''My configuration class.'''
            ...     pass
            >>> MyConfig().desc
            'My configuration class.'
        """
        desc = re.split(pattern=r"\w+\n\s{4}\-{3,}", string=self.__doc__, maxsplit=0)[0]
        return inspect.cleandoc(desc)

    @property
    def _repr_content(self) -> dict:
        """The items that are displayed in the __repr__ method.

        This property can be overridden in order to modify the output of the __repr__ method.

        Returns:
            dict: A dictionary containing the items to be displayed in the __repr__ method.

        Examples:
            >>> class MyConfig(Config):
            ...     '''My configuration class.'''
            ...     pass
            >>> MyConfig()._repr_content
            {'Name': 'MyConfig'}
        """
        content = {}
        content["Name"] = self.__class__.__name__
        return content


class Dataset(abc.ABC):
    """Base class for all datasets.

    All datasets inherit from this class, be they stored in a file or generated on the fly.

    Args:
        task (str): Type of task the dataset is meant for. Should be one of:
            - "Regression"
            - "Binary classification"
            - "Multi-class classification"
            - "Multi-output binary classification"
            - "Multi-output regression"
        n_features (int): Number of features in the dataset.
        n_samples (int, optional): Number of samples in the dataset.
        n_classes (int, optional): Number of classes in the dataset, only applies to classification datasets.
        n_outputs (int, optional): Number of outputs the target is made of, only applies to multi-output datasets.
        sparse (bool, optional): Whether the dataset is sparse or not.

    Attributes:
        desc (str): The description from the docstring.
        _repr_content (dict): The items that are displayed in the __repr__ method.
    """

    def __init__(
        self,
        task: str,
        n_features: int,
        n_samples: Optional[int] = None,
        n_classes: Optional[int] = None,
        n_outputs: Optional[int] = None,
        sparse: bool = False,
    ):
        """Initialize a Dataset object.

        Args:
            task (str): Type of task the dataset is meant for. Should be one of:
                - "Regression"
                - "Binary classification"
                - "Multi-class classification"
                - "Multi-output binary classification"
                - "Multi-output regression"
            n_features (int): Number of features in the dataset.
            n_samples (int, optional): Number of samples in the dataset. Defaults to None.
            n_classes (int, optional): Number of classes in the dataset, only applies to classification datasets.
                Defaults to None.
            n_outputs (int, optional): Number of outputs the target is made of, only applies to multi-output datasets.
                Defaults to None.
            sparse (bool, optional): Whether the dataset is sparse or not. Defaults to False.
        """
        self.task = task
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.n_classes = n_classes
        self.sparse = sparse

    @abc.abstractmethod
    def __iter__(self):
        """Abstract method for iterating over samples in the dataset."""
        raise NotImplementedError

    def take(self, k: int) -> itertools.islice:
        """Iterate over the k samples.

        Args:
            k (int): The number of samples to iterate over.

        Returns:
            itertools.islice: An iterator over the first k samples in the dataset.

        Examples:
            >>> class MyDataset(Dataset):
            ...     def __init__(self):
            ...         super().__init__('Regression', 10)
            ...     def __iter__(self):
            ...         yield from range(10)
            >>> list(MyDataset().take(5))
            [0, 1, 2, 3, 4]
        """
        return itertools.islice(self, k)

    @property
    def desc(self) -> str:
        """Return the description from the docstring.

        Returns:
            str: The description from the docstring.

        Examples:
            >>> class MyDataset(Dataset):
            ...     '''My dataset class.'''
            ...     def __init__(self):
            ...         super().__init__('Regression', 10)
            ...     def __iter__(self):
            ...         yield from range(10)
            >>> MyDataset().desc
            'My dataset class.'
        """
        desc = re.split(pattern=r"\w+\n\s{4}\-{3,}", string=self.__doc__, maxsplit=0)[0]
        return inspect.cleandoc(desc)

    @property
    def _repr_content(self) -> dict:
        """The items that are displayed in the __repr__ method.

        This property can be overridden in order to modify the output of the __repr__ method.

        Returns:
            dict: A dictionary containing the items to be displayed in the __repr__ method.
        """

        content = {}
        content["Name"] = self.__class__.__name__
        content["Task"] = self.task
        if isinstance(self, SyntheticDataset) and self.n_samples is None:
            content["Samples"] = "∞"
        elif self.n_samples:
            content["Samples"] = f"{self.n_samples:,}"
        if self.n_features:
            content["Features"] = f"{self.n_features:,}"
        if self.n_outputs:
            content["Outputs"] = f"{self.n_outputs:,}"
        if self.n_classes:
            content["Classes"] = f"{self.n_classes:,}"
        content["Sparse"] = str(self.sparse)

        return content

    def __repr__(self):
        l_len = max(map(len, self._repr_content.keys()))
        r_len = max(map(len, self._repr_content.values()))

        out = f"{self.desc}\n\n" + "\n".join(k.rjust(l_len) + "  " + v.ljust(r_len) for k, v in self._repr_content.items())

        if "Parameters\n    ----------" in self.__doc__:
            params = re.split(
                r"\w+\n\s{4}\-{3,}",
                re.split("Parameters\n    ----------", self.__doc__)[1],
            )[0].rstrip()
            out += f"\n\nParameters\n----------{params}"

        return out


class SyntheticDataset(Dataset):
    """A synthetic dataset.

    Args:
        task (str): Type of task the dataset is meant for. Should be one of:
            - "Regression"
            - "Binary classification"
            - "Multi-class classification"
            - "Multi-output binary classification"
            - "Multi-output regression"
        n_features (int): Number of features in the dataset.
        n_samples (int): Number of samples in the dataset.
        n_classes (int): Number of classes in the dataset, only applies to classification datasets.
        n_outputs (int): Number of outputs the target is made of, only applies to multi-output datasets.
        sparse (bool): Whether the dataset is sparse or not.

    Returns:
        (SyntheticDataset): A synthetic dataset object.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_features=4, random_state=0)
        >>> dataset = SyntheticDataset(task="Binary classification",
                                            n_features=4,
                                            n_samples=100,
                                            n_classes=2,
                                            n_outputs=1,
                                            sparse=False)

    """

    def __init__(
        self,
        task: str,
        n_features: int,
        n_samples: int,
        n_classes: Union[int, None] = None,
        n_outputs: Union[int, None] = None,
        sparse: bool = False,
    ):
        pass

    def __repr__(self) -> str:
        """String representation of the SyntheticDataset object.

        Returns:
            str: A string representation of the SyntheticDataset object.

        Examples:
            >>> from sklearn.datasets import make_classification
            >>> X, y = make_classification(n_features=4, random_state=0)
            >>> dataset = SyntheticDataset(task="Binary classification",
                                                n_features=4,
                                                n_samples=100,
                                                n_classes=2,
                                                n_outputs=1,
                                                sparse=False)
            >>> print(dataset)
            Synthetic data generator

            Configuration
            -------------
                task  Binary classification
          n_features  4
           n_samples  100
           n_classes  2
           n_outputs  1
              sparse  False
        """
        l_len_prop = max(map(len, self._repr_content.keys()))
        r_len_prop = max(map(len, self._repr_content.values()))
        params = self._get_params()
        l_len_config = max(map(len, params.keys()))
        r_len_config = max(map(len, map(str, params.values())))

        out = (
            "Synthetic data generator\n\n"
            + "\n".join(k.rjust(l_len_prop) + "  " + v.ljust(r_len_prop) for k, v in self._repr_content.items())
            + "\n\nConfiguration\n-------------\n"
            + "\n".join(k.rjust(l_len_config) + "  " + str(v).ljust(r_len_config) for k, v in params.items())
        )

        return out

    def _get_params(self) -> typing.Dict[str, typing.Any]:
        """Return the parameters that were used during initialization.

        Returns:
            dict: A dictionary containing the parameters that were used during initialization.

        Examples:
            >>> from sklearn.datasets import make_classification
            >>> X, y = make_classification(n_features=4, random_state=0)
            >>> dataset = SyntheticDataset(task="Binary classification",
                                            n_features=4,
                                            n_samples=100,
                                            n_classes=2,
                                            n_outputs=1,
                                            sparse=False)
            >>> dataset._get_params()
            {'task': 'Binary classification',
             'n_features': 4,
             'n_samples': 100,
             'n_classes': 2,
             'n_outputs': 1,
             'sparse': False}
        """
        return {name: getattr(self, name) for name, param in inspect.signature(self.__init__).parameters.items() if param.kind != param.VAR_KEYWORD}  # type: ignore


class FileConfig(Config):
    """Base class for configurations that are stored in a local file.

    Args:
        filename (str): The file's name.
        directory (Optional[str]):
            The directory where the file is contained.
            Defaults to the location of the `datasets` module.
        desc (dict): Extra config parameters to pass as keyword arguments.

    Returns:
        (FileConfig): A FileConfig object.

    Examples:
        >>> config = FileConfig(filename="config.json", directory="/path/to/directory")
    """

    def __init__(self, filename: str, directory: Optional[str] = None, **desc):
        super().__init__(**desc)
        self.filename = filename
        self.directory = directory

    @property
    def path(self) -> pathlib.Path:
        """The path to the configuration file.

        Returns:
            pathlib.Path: The path to the configuration file.

        Examples:
            >>> config = FileConfig(filename="config.json", directory="/path/to/directory")
            >>> config.path
            PosixPath('/path/to/directory/config.json')
        """
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    @property
    def _repr_content(self) -> dict:
        """The content of the string representation of the FileConfig object.

        Returns:
            dict: A dictionary containing the content of the string representation of the FileConfig object.

        Examples:
            >>> config = FileConfig(filename="config.json", directory="/path/to/directory")
            >>> config._repr_content
            {'Path': '/path/to/directory/config.json'}
        """
        content = super()._repr_content
        content["Path"] = str(self.path)
        return content


class FileDataset(Dataset):
    """Base class for datasets that are stored in a local file.

    Small datasets that are part of the spotriver package inherit from this class.

    Args:
        filename (str): The file's name.
        directory (Optional[str]):
            The directory where the file is contained.
            Defaults to the location of the `datasets` module.
        desc (dict): Extra dataset parameters to pass as keyword arguments.

    Returns:
        (FileDataset): A FileDataset object.

    Examples:
        >>> dataset = FileDataset(filename="dataset.csv", directory="/path/to/directory")
    """

    def __init__(self, filename: str, directory: Optional[str] = None, **desc):
        super().__init__(**desc)
        self.filename = filename
        self.directory = directory

    @property
    def path(self) -> pathlib.Path:
        """The path to the dataset file.

        Returns:
            pathlib.Path: The path to the dataset file.

        Examples:
            >>> dataset = FileDataset(filename="dataset.csv", directory="/path/to/directory")
            >>> dataset.path
            PosixPath('/path/to/directory/dataset.csv')
        """
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    @property
    def _repr_content(self) -> dict:
        """The content of the string representation of the FileDataset object.

        Returns:
            dict: A dictionary containing the content of the string representation of the FileDataset object.

        Examples:
            >>> dataset = FileDataset(filename="dataset.csv", directory="/path/to/directory")
            >>> dataset._repr_content
            {'Path': '/path/to/directory/dataset.csv'}
        """
        content = super()._repr_content
        content["Path"] = str(self.path)
        return content


class RemoteDataset(FileDataset):
    """Base class for datasets that are stored in a remote file.

    Medium and large datasets that are not part of the river package inherit from this class.

    The filename doesn't have to be provided if unpack is False. Indeed in the latter case the
    filename will be inferred from the URL.

    Args:
        url (str): The URL the dataset is located at.
        size (int): The expected download size.
        unpack (bool): Whether to unpack the download or not. Defaults to True.
        filename (str):
            An optional name to given to the file if the file is unpacked. Defaults to None.
        desc (dict): Extra dataset parameters to pass as keyword arguments.

    Examples:

        >>> from river.datasets import AirlinePassengers
        >>> dataset = AirlinePassengers()
        >>> for x, y in dataset:
        ...     print(x, y)
        ...     break
        ({'month': datetime.datetime(1949, 1, 1, 0, 0)}, 112)

    """

    def __init__(self, url: str, size: int, unpack: bool = True, filename: str = None, **desc: dict):
        if filename is None:
            filename = path.basename(url)

        super().__init__(filename=filename, **desc)
        self.url = url
        self.size = size
        self.unpack = unpack

    @property
    def path(self) -> pathlib.Path:
        """Returns the path where the dataset is stored."""
        return pathlib.Path(get_data_home(), self.__class__.__name__, self.filename)

    def download(self, force: bool = False, verbose: bool = True) -> None:
        """Downloads the dataset.

        Args:
            force (bool):
                Whether to force the download even if the data is already downloaded.
                Defaults to False.
            verbose (bool):
                Whether to display information about the download. Defaults to True.

        """
        if not force and self.is_downloaded:
            return

        # Determine where to download the archive
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        archive_path = directory.joinpath(path.basename(self.url))

        with request.urlopen(self.url) as r:
            # Notify the user
            if verbose:
                meta = r.info()
                try:
                    n_bytes = int(meta["Content-Length"])
                    msg = f"Downloading {self.url} ({n_bytes})"
                except KeyError:
                    msg = f"Downloading {self.url}"
                print(msg)

            # Now dump the contents of the requests
            with open(archive_path, "wb") as f:
                shutil.copyfileobj(r, f)

        if not self.unpack:
            return

        if verbose:
            print(f"Uncompressing into {directory}")

        if archive_path.suffix.endswith("zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(directory)

        elif archive_path.suffix.endswith(("gz", "tar")):
            mode = "r:" if archive_path.suffix.endswith("tar") else "r:gz"
            tar = tarfile.open(archive_path, mode)
            tar.extractall(directory)
            tar.close()

        else:
            raise RuntimeError(f"Unhandled extension type: {archive_path.suffix}")

        # Delete the archive file now that it has been uncompressed
        archive_path.unlink()

    @abc.abstractmethod
    def _iter(self):
        pass

    @property
    def is_downloaded(self) -> bool:
        """Indicate whether or not the data has been correctly downloaded."""
        if self.path.exists():
            if self.path.is_file():
                return self.path.stat().st_size == self.size
            return sum(f.stat().st_size for f in self.path.glob("**/*") if f.is_file())

        return False

    def __iter__(self):
        """Iterates over the samples of a dataset."""
        if not self.is_downloaded:
            self.download(verbose=True)
        if not self.is_downloaded:
            raise RuntimeError("Something went wrong during the download")
        yield from self._iter()

    @property
    def _repr_content(self):
        content = super()._repr_content
        content["URL"] = self.url
        content["Size"] = self.size
        content["Downloaded"] = str(self.is_downloaded)
        return content


class GenericFileDataset(Dataset):
    """Base class for datasets that are stored in a local file.

    Small datasets that are part of the spotriver package inherit from this class.

    Args:
        filename (str): The file's name.
        target (str): The name of the target variable.
        converters (dict):
            A dictionary specifying how to convert the columns of the dataset. Defaults to None.
        parse_dates (list): A list of columns to parse as dates. Defaults to None.
        directory (str):
            The directory where the file is contained. Defaults to the location of the `datasets` module.
        desc (dict): Extra dataset parameters to pass as keyword arguments.


    Examples:

        >>> from river.datasets import Iris
        >>> dataset = Iris()
        >>> for x, y in dataset:
        ...     print(x, y)
        ...     break
        ({'sepal_length': 5.1,
          'sepal_width': 3.5,
          'petal_length': 1.4,
          'petal_width': 0.2},
          'setosa')

    """

    def __init__(
        self,
        filename: str,
        target: str,
        converters: dict = None,
        parse_dates: list = None,
        directory: str = None,
        **desc: dict,
    ):
        super().__init__(**desc)
        self.filename = filename
        self.directory = directory
        self.target = target
        self.converters = converters
        self.parse_dates = parse_dates

    @property
    def path(self) -> pathlib.Path:
        """Returns the path where the dataset is stored."""
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    @property
    def _repr_content(self):
        content = super()._repr_content
        content["Path"] = str(self.path)
        return content
