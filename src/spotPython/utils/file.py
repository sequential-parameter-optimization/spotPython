import torchvision
import torchvision.transforms as transforms
import pickle
from spotPython.utils.init import (
    design_control_init,
    surrogate_control_init,
    optimizer_control_init,
)

# from torch.utils.tensorboard import SummaryWriter


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


def save_experiment(
    spot_tuner, fun_control, design_control=None, surrogate_control=None, optimizer_control=None
) -> str:
    """
    Saves the experiment as a pickle file.

    Args:
        spot_tuner (object): The spot tuner object.
        fun_control (dict): The function control dictionary.
        design_control (dict, optional): The design control dictionary. Defaults to None.
        surrogate_control (dict, optional): The surrogate control dictionary. Defaults to None.
        optimizer_control (dict, optional): The optimizer control dictionary. Defaults to None.

    Returns:
        PKL_NAME (str):
            Name of the pickle file. Build as "spot_" + PREFIX + "experiment.pickle".

    """
    if design_control is None:
        design_control = design_control_init()
    if surrogate_control is None:
        surrogate_control = surrogate_control_init()
    if optimizer_control is None:
        optimizer_control = optimizer_control_init()
    # remove the key "spot_writer" from the fun_control dictionary,
    # because it is not serializable.
    # TODO: It will be re-added when the experiment is loaded.
    fun_control.pop("spot_writer", None)

    experiment = {
        "spot_tuner": spot_tuner,
        "fun_control": fun_control,
        "design_control": design_control,
        "surrogate_control": surrogate_control,
        "optimizer_control": optimizer_control,
    }
    PREFIX = fun_control["PREFIX"]
    PKL_NAME = "spot_" + PREFIX + "experiment.pickle"
    with open(PKL_NAME, "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Experiment saved as {PKL_NAME}")
    return PKL_NAME


def load_experiment(PKL_NAME):
    """
    Loads the experiment from a pickle file.

    Args:
        PKL_NAME (str): Name of the pickle file.

    Returns:
        spot_tuner (object): The spot tuner object.
        fun_control (dict): The function control dictionary.
        design_control (dict): The design control dictionary.
        surrogate_control (dict): The surrogate control dictionary.
        optimizer_control (dict): The optimizer control dictionary.

    """
    with open(PKL_NAME, "rb") as handle:
        experiment = pickle.load(handle)
    spot_tuner = experiment["spot_tuner"]
    fun_control = experiment["fun_control"]
    design_control = experiment["design_control"]
    surrogate_control = experiment["surrogate_control"]
    optimizer_control = experiment["optimizer_control"]
    # TODO: Add the key "spot_writer" to the fun_control dictionary,
    # because it was not saved in the pickle file.
    return spot_tuner, fun_control, design_control, surrogate_control, optimizer_control
