import torchvision
import torchvision.transforms as transforms
import pickle
import os
import json
import sys
import importlib

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


def get_experiment_filename(PREFIX):
    """Returns the name of the experiment.

    Args:
        PREFIX (str): Prefix of the experiment.

    Returns:
        filename (str): Name of the experiment.

    Examples:
        >>> from spotPython.utils.file import get_experiment_name
        >>> from spotPython.utils.init import fun_control_init
        >>> fun_control = fun_control_init(PREFIX="branin")
        >>> PREFIX = fun_control["PREFIX"]
        >>> filename = get_experiment_filename(PREFIX)
    """
    if PREFIX is None:
        return None
    else:
        filename = "spot_" + PREFIX + "_experiment.pickle"
    return filename


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
    return spot_tuner, fun_control, design_control, surrogate_control, optimizer_control


def load_dict_from_file(coremodel, dirname="userModel"):
    """Loads a dictionary from a json file.

    Args:
        coremodel (str): Name of the core model.
        dirname (str, optional): Directory name. Defaults to "userModel".

    Returns:
        dict (dict): Dictionary with the core model.

    """
    file_path = os.path.join(dirname, f"{coremodel}.json")
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            dict_tmp = json.load(f)
            dict = dict_tmp[coremodel]
    else:
        print(f"The file {file_path} does not exist.")
        dict = None
    return dict


def load_core_model_from_file(coremodel, dirname="userModel"):
    """Loads a core model from a python file.

    Args:
        coremodel (str): Name of the core model.
        dirname (str, optional): Directory name. Defaults to "userModel".

    Returns:
        coremodel (object): Core model.

    """
    sys.path.insert(0, "./" + dirname)
    module = importlib.import_module(coremodel)
    core_model = getattr(module, coremodel)
    return core_model
