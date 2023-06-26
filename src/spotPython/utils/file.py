import torchvision
import torchvision.transforms as transforms
import socket
from datetime import datetime
from dateutil.tz import tzlocal


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
