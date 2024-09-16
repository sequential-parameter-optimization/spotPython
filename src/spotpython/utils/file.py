import torchvision
import torchvision.transforms as transforms
import pickle
import os
import json
import sys
import importlib
from spotpython.hyperparameters.values import get_tuned_architecture
from spotpython.utils.eda import gen_design_table
from spotpython.utils.init import setup_paths


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


def get_experiment_filename(PREFIX):
    """Returns the name of the experiment.

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
        filename = "spot_" + PREFIX + "_experiment.pickle"
    return filename


def load_experiment(PKL_NAME):
    """
    Loads the experiment from a pickle file.
    If the spot tuner object and the fun control dictionary do not exist, an error is thrown.
    If the design control, surrogate control, and optimizer control dictionaries do not exist, a warning is issued
    and `None` is assigned to the corresponding variables.

    Args:
        PKL_NAME (str): Name of the pickle file.

    Returns:
        spot_tuner (object): The spot tuner object.
        fun_control (dict): The function control dictionary.
        design_control (dict): The design control dictionary.
        surrogate_control (dict): The surrogate control dictionary.
        optimizer_control (dict): The optimizer control dictionary.

    Notes:
        The corresponding save_experiment function is part of the class spot.

    Examples:
        >>> from spotpython.utils.file import load_experiment
        >>> spot_tuner, fun_control, design_control, _, _ = load_experiment("spot_0_experiment.pickle")

    """
    with open(PKL_NAME, "rb") as handle:
        experiment = pickle.load(handle)
    # assign spot_tuner and fun_control only if they exist otherwise throw an error
    if "spot_tuner" not in experiment:
        raise ValueError("The spot tuner object does not exist in the pickle file.")
    if "fun_control" not in experiment:
        raise ValueError("The fun control dictionary does not exist in the pickle file.")
    spot_tuner = experiment["spot_tuner"]
    fun_control = experiment["fun_control"]
    # assign the rest of the dictionaries if they exist otherwise assign None
    if "design_control" not in experiment:
        design_control = None
        # issue a warning
        print("The design control dictionary does not exist in the pickle file. Returning None.")
    else:
        design_control = experiment["design_control"]
    if "surrogate_control" not in experiment:
        surrogate_control = None
        # issue a warning
        print("The surrogate control dictionary does not exist in the pickle file. Returning None.")
    else:
        surrogate_control = experiment["surrogate_control"]
    if "optimizer_control" not in experiment:
        # issue a warning
        print("The optimizer control dictionary does not exist in the pickle file. Returning None.")
        optimizer_control = None
    else:
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


def get_experiment_from_PREFIX(PREFIX, return_dict=True) -> dict:
    """
    Setup the experiment based on the PREFIX provided and return the relevant configuration
    and control objects.

    Args:
        PREFIX (str):
            The prefix for the experiment filename.
        return_dict (bool, optional):
            Whether to return the configuration and control objects as a dictionary.
            If False, a tuple is returned:
            "(config, fun_control, design_control, surrogate_control, optimizer_control)."
            Defaults to True.

    Returns:
        dict: Dictionary containing the configuration and control objects.

    Example:
        >>> from spotpython.utils.file import get_experiment_from_PREFIX
        >>> config = get_experiment_from_PREFIX("100")["config"]

    """
    experiment_name = get_experiment_filename(PREFIX)
    spot_tuner, fun_control, design_control, surrogate_control, optimizer_control = load_experiment(experiment_name)
    config = get_tuned_architecture(spot_tuner, fun_control)
    if return_dict:
        return {
            "config": config,
            "fun_control": fun_control,
            "design_control": design_control,
            "surrogate_control": surrogate_control,
            "optimizer_control": optimizer_control,
        }
    else:
        return config, fun_control, design_control, surrogate_control, optimizer_control


def load_and_run_spot_python_experiment(spot_pkl_name) -> tuple:
    """Loads and runs a spot experiment.

    Args:
        spot_pkl_name (str):
            The name of the spot experiment file.

    Returns:
        tuple: A tuple containing the spot tuner, fun control,
               design control, surrogate control, optimizer control,
               and the tensorboard process object (p_popen).

    Notes:
        p_open is deprecated and should be removed in future versions.
        It returns None.

    Examples:
        >>> from spotpython.utils.file import load_and_run_spot_python_experiment
        >>> spot_tuner = load_and_run_spot_python_experiment("spot_branin_experiment.pickle")

    """
    p_open = None
    (spot_tuner, fun_control, design_control, surrogate_control, optimizer_control) = load_experiment(spot_pkl_name)
    print("\nLoaded fun_control in spotRun():")
    # pprint.pprint(fun_control)
    print(gen_design_table(fun_control))
    setup_paths(fun_control["TENSORBOARD_CLEAN"])
    spot_tuner.init_spot_writer()
    # if fun_control["tensorboard_start"]:
    #     p_open = start_tensorboard()
    # else:
    #     p_open = None
    spot_tuner.run()
    # # tensorboard --logdir="runs/"
    # stop_tensorboard(p_open)
    print(gen_design_table(fun_control=fun_control, spot=spot_tuner))
    return spot_tuner, fun_control, design_control, surrogate_control, optimizer_control, p_open
