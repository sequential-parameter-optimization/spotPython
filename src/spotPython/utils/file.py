import torchvision
import torchvision.transforms as transforms
import socket
from datetime import datetime
from dateutil.tz import tzlocal
import pickle
import os


def load_data(data_dir="./data"):
    """Loads the CIFAR10 dataset.
    Args:
        data_dir (str, optional): Directory to save the data. Defaults to "./data".
    Returns:
        trainset (torchvision.datasets.CIFAR10): Training dataset.
    Examples:
        >>> from spotPython.utils.file import load_data
        >>> trainset = load_data(data_dir="./data")

    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


def get_experiment_name(prefix: str = "00") -> str:
    """Returns a unique experiment name with a given prefix.
    Args:
        prefix (str, optional): Prefix for the experiment name. Defaults to "00".
    Returns:
        str: Unique experiment name.
    Examples:
        >>> from spotPython.utils.file import get_experiment_name
        >>> get_experiment_name(prefix="00")
        00_ubuntu_2021-08-31_14-30-00
    """
    start_time = datetime.now(tzlocal())
    HOSTNAME = socket.gethostname().split(".")[0]
    experiment_name = prefix + "_" + HOSTNAME + "_" + str(start_time).split(".", 1)[0].replace(" ", "_")
    experiment_name = experiment_name.replace(":", "-")
    return experiment_name


def save_pickle(obj, filename: str):
    """Saves an object as a pickle file.
        Add .pkl to the filename.
    Args:
        obj (object): Object to be saved.
        filename (str): Name of the pickle file.
    Examples:
        >>> from spotPython.utils.file import save_pickle
        >>> save_pickle(obj, filename="obj.pkl")
    """
    filename = filename + ".pkl"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename: str):
    """Loads a pickle file.
        Add .pkl to the filename.
    Args:
        filename (str): Name of the pickle file.
    Returns:
        (object): Loaded object.
    Examples:
        >>> from spotPython.utils.file import load_pickle
        >>> obj = load_pickle(filename="obj.pkl")
    """
    filename = filename + ".pkl"
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_spot_tensorboard_path(experiment_name):
    """Get the path to the spot tensorboard files.
    Args:
        experiment_name (str): The name of the experiment.
    Returns:
        spot_tensorboard_path (str): The path to the folder where the spot tensorboard files are saved.
    """
    spot_tensorboard_path = os.environ.get("PATH_TENSORBOARD", "runs/spot_logs/")
    spot_tensorboard_path = os.path.join(spot_tensorboard_path, experiment_name)
    return spot_tensorboard_path


def get_tensorboard_path(fun_control):
    """Get the path to the tensorboard files.
    Args:
        fun_control (dict): The function control dictionary.
    Returns:
        tensorboard_path (str): The path to the folder where the tensorboard files are saved.
    """
    return fun_control["TENSORBOARD_PATH"]
