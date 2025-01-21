import torchvision
import torchvision.transforms as transforms
import pickle
import os
import json
import sys
import importlib


# from torch.utils.tensorboard import SummaryWriter


def load_cifar10_data(data_dir="./data"):
    """Loads the CIFAR10 dataset.

    Args:
        data_dir (str, optional): Directory to save the data. Defaults to "./data".

    Returns:
        trainset (torchvision.datasets.CIFAR10): Training dataset.

    Examples:
        >>> from spotpython.utils.file import load_cifar10_data
        >>> trainset = load_cifar10_data(data_dir="./data")

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
        >>> from spotpython.utils.file import save_pickle
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
        >>> from spotpython.utils.file import load_pickle
        >>> obj = load_pickle(filename="obj.pkl")
    """
    filename = filename + ".pkl"
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_experiment_filename(PREFIX) -> str:
    """Returns the name of the experiment file.
    This is the PREFIX with the suffix "_exp.pkl".
    It is none, if PREFIX is None.

    Args:
        PREFIX (str): Prefix of the experiment.

    Returns:
        filename (str): Name of the experiment.

    Examples:
        >>> from spotpython.utils.file import get_experiment_name
        >>> from spotpython.utils.init import fun_control_init
        >>> fun_control = fun_control_init(PREFIX="branin")
        >>> PREFIX = fun_control["PREFIX"]
        >>> filename = get_experiment_filename(PREFIX)
    """
    if PREFIX is None:
        return None
    else:
        filename = PREFIX + "_exp.pkl"
    return filename


def get_result_filename(PREFIX) -> str:
    """Returns the name of the result file.
    This is the PREFIX with the suffix "_res.pkl".
    It is none, if PREFIX is None.

    Args:
        PREFIX (str): Prefix of the experiment.

    Returns:
        filename (str): Name of the experiment.

    Examples:
        >>> from spotpython.utils.file import get_experiment_name
        >>> from spotpython.utils.init import fun_control_init
        >>> fun_control = fun_control_init(PREFIX="branin")
        >>> PREFIX = fun_control["PREFIX"]
        >>> filename = get_experiment_filename(PREFIX)
    """
    if PREFIX is None:
        return None
    else:
        filename = PREFIX + "_res.pkl"
    return filename


def _handle_res_filename(filename, PREFIX):
    if filename is None:
        if PREFIX is None:
            raise ValueError("No PREFIX provided.")
        filename = get_result_filename(PREFIX)
    return filename


def _handle_exp_filename(filename, PREFIX):
    if filename is None:
        if PREFIX is None:
            raise ValueError("No PREFIX provided.")
        filename = get_experiment_filename(PREFIX)
    return filename


def load_result(PREFIX=None, filename=None) -> tuple:
    """Loads the result from a pickle file with the name
    PREFIX + "_res.pkl".
    This is the standard filename for the result file,
    when it is saved by the spot tuner using `save_result()`, i.e.,
    when fun_control["save_result"] is set to True.
    If a filename is provided, the result is loaded from this file.

    Args:
        PREFIX (str): Prefix of the experiment. Defaults to None.
        filename (str): Name of the pickle file. Defaults to None.

    Returns:
        spot_tuner (Spot): The spot tuner object.

    Notes:
        The corresponding save_result function is part of the class spot.

    Examples:
        >>> from spotpython.utils.file import load_result
        >>> load_result("branin")

    """
    filename = _handle_res_filename(filename, PREFIX)
    spot_tuner = load_experiment(filename=filename)
    return spot_tuner


def load_experiment(PREFIX=None, filename=None):
    """
    Loads the experiment from a pickle file.
    If filename is None and PREFIX is not None, the experiment is loaded based on the PREFIX
    using the get_experiment_filename function.
    If the spot tuner object and the fun control dictionary do not exist, an error is thrown.
    If the design control, surrogate control, and optimizer control dictionaries do not exist, a warning is issued
    and `None` is assigned to the corresponding variables.

    Args:
        PREFIX (str, optional): Prefix of the experiment. Defaults to None.
        filename (str): Name of the pickle file. Defaults to None.

    Returns:
        spot_tuner (Spot): The spot tuner object.

    Notes:
        The corresponding save_experiment function is part of the class spot.

    Examples:
        >>> from spotpython.utils.file import load_experiment
        >>> spot_tuner, fun_control, design_control, _, _ = load_experiment(filename="RUN_0.pkl")

    """
    filename = _handle_exp_filename(filename, PREFIX)
    with open(filename, "rb") as handle:
        spot_tuner = pickle.load(handle)
        print(f"Loaded experiment from {filename}")
    return spot_tuner


def load_and_run_spot_python_experiment(PREFIX=None, filename=None) -> object:
    """Loads and runs a spot experiment.

    Args:
        PREFIX (str, optional): Prefix of the experiment. Defaults to None.
        filename (str): Name of the pickle file. Defaults to None

    Returns:
        spot_tuner (Spot): The spot tuner object.

    Examples:
        >>> from spotpython.utils.file import load_and_run_spot_python_experiment
        >>> spot_tuner = load_and_run_spot_python_experiment(filename="spot_branin_experiment.pickle")
        >>> # Or use PREFIX
        >>> spot_tuner = load_and_run_spot_python_experiment(PREFIX="spot_branin_experiment")

    """
    S = load_experiment(PREFIX=PREFIX, filename=filename)
    S.run()
    return S


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
