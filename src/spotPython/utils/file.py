import torchvision
import torchvision.transforms as transforms
import pickle
from spotPython.utils.init import (
    design_control_init,
    surrogate_control_init,
    optimizer_control_init,
)
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
            Name of the pickle file. Build as "spot_" + PREFIX + "_experiment.pickle".

    Examples:
        >>> import os
            from spotPython.utils.file import save_experiment, load_experiment
            import numpy as np
            from math import inf
            from spotPython.spot import spot
            from spotPython.utils.init import (
                fun_control_init,
                design_control_init,
                surrogate_control_init,
                optimizer_control_init)
            from spotPython.fun.objectivefunctions import analytical
                fun = analytical().fun_branin
            fun_control = fun_control_init(
                        PREFIX="branin",
                        SUMMARY_WRITER=False,
                        lower = np.array([0, 0]),
                        upper = np.array([10, 10]),
                        fun_evals=8,
                        fun_repeats=1,
                        max_time=inf,
                        noise=False,
                        tolerance_x=0,
                        ocba_delta=0,
                        var_type=["num", "num"],
                        infill_criterion="ei",
                        n_points=1,
                        seed=123,
                        log_level=20,
                        show_models=False,
                        show_progress=True)
            design_control = design_control_init(
                        init_size=5,
                        repeats=1)
            surrogate_control = surrogate_control_init(
                        model_fun_evals=10000,
                        min_theta=-3,
                        max_theta=3,
                        n_theta=2,
                        theta_init_zero=True,
                        n_p=1,
                        optim_p=False,
                        var_type=["num", "num"],
                        seed=124)
            optimizer_control = optimizer_control_init(
                        max_iter=1000,
                        seed=125)
            spot_tuner = spot.Spot(fun=fun,
                        fun_control=fun_control,
                        design_control=design_control,
                        surrogate_control=surrogate_control,
                        optimizer_control=optimizer_control)
            # Call the save_experiment function
            pkl_name = save_experiment(
                spot_tuner=spot_tuner,
                fun_control=fun_control,
                design_control=None,
                surrogate_control=None,
                optimizer_control=None
            )
            # Call the load_experiment function
            (spot_tuner_1, fun_control_1, design_control_1,
                surrogate_control_1, optimizer_control_1) = load_experiment(pkl_name)
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
    # check if the key "spot_writer" is in the fun_control dictionary
    if "spot_writer" in fun_control and fun_control["spot_writer"] is not None:
        fun_control["spot_writer"].close()
    PREFIX = fun_control["PREFIX"]
    PKL_NAME = "spot_" + PREFIX + "_experiment.pickle"
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
