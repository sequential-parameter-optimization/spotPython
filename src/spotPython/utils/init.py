import os
import lightning as L
import datetime


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter


def fun_control_init(
    _L_in=None,
    _L_out=None,
    TENSORBOARD_CLEAN=False,
    accelerator="auto",
    device=None,
    devices=1,
    enable_progress_bar=False,
    fun_evals=15,
    fun_repeats=1,
    log_level=50,
    max_time=1,
    noise=False,
    num_workers=0,
    seed=1234,
    sigma=0.0,
    show_progress=False,
    spot_tensorboard_path=None,
    task=None,
    test_seed=1234,
    test_size=0.4,
    tolerance_x=0,
):
    """Initialize fun_control dictionary.
    Args:
        task (str):
            The task to perform. It can be either "classification" or "regression".
        _L_in (int):
            The number of input features.
        _L_out (int):
            The number of output features.
        acceleration (str):
            The accelerator to be used by the Lighting Trainer.
            It can be either "auto", "dp", "ddp", "ddp2", "ddp_spawn", "ddp_cpu", "gpu", "tpu".
        counter (int):
            The counter for the number of function evaluations. Updated in
            Spot update_stats(). Initialized to 0.
        device (str):
            The device to use for the training. It can be either "cpu", "mps", or "cuda".
        devices (str or int):
            The number of devices to use for the training/validation/testing.
            Default is 1. Can be "auto" or an integer.
        enable_progress_bar (bool):
            Whether to enable the progress bar or not.
        fun_evals (int):
            The number of function evaluations.
        fun_repeats (int):
            The number of function repeats. Default is 1.
        log_level (int):
            The log level. Default is 50 (ERROR).
        max_time (int):
            The maximum time in minutes.
        noise (bool):
            Whether the objective function is noiy or not. Default is False.
            Affects the repeat of the function evaluations.
        num_workers (int):
            The number of workers to use for the data loading. Default is 0.
        seed (int):
            The seed to use for the random number generator.
        sigma (float):
            The standard deviation of the noise of the objective function.
        show_progress (bool):
            Whether to show the progress or not. Default is False.
        spot_tensorboard_path (str):
            The path to the folder where the spot tensorboard files are saved.
            If None, no spot tensorboard files are saved.
        task (str):
            The task to perform. It can be either "classification" or "regression".
        test_seed (int):
            The seed to use for the test set. Default is 1234.
        test_size (float):
            The size of the test set. Default is 0.4, i.e.,
            60% of the data is used for training and 40% for testing.
        tolerance_x (float):
            The tolerance for the new x values. Default is 0.

    Returns:
        fun_control (dict):
            A dictionary containing the information about the core model,
            loss function, metrics, and the hyperparameters.

    Examples:
        >>> from spotPy.utils.init import fun_control_init
            fun_control = fun_control_init(_L_in=64, _L_out=11, num_workers=0, device=None)
            fun_control
            {'CHECKPOINT_PATH': 'saved_models/',
                'DATASET_PATH': 'data/',
                'RESULTS_PATH': 'results/',
                'TENSORBOARD_PATH': 'runs/',
                '_L_in': 64,
                '_L_out': 11,
                'accelerator': "auto",
                'data': None,
                'data_dir': './data',
                'device': None,
                'devices': "auto",
                'enable_progress_bar': False,
                'eval': None,
                'k_folds': None,
                'loss_function': None,
                'metric_river': None,
                'metric_sklearn': None,
                'metric_torch': None,
                'metric_params': {},
                'model_dict': {},
                'noise': False,
                'n_samples': None,
                'num_workers': 0,
                'optimizer': None,
                'path': None,
                'prep_model': None,
                'save_model': False,
                'seed': 1234,
                'show_batch_interval': 1000000,
                'shuffle': None,
                'sigma': 0.0,
                'target_column': None,
                'train': None,
                'test': None,
                'task': 'classification',
                'tensorboard_path': None,
                'weights': 1.0,
                'writer': None}
    """
    # Setting the seed
    L.seed_everything(seed)

    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "runs/saved_models/")
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    # Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
    DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
    os.makedirs(DATASET_PATH, exist_ok=True)
    # Path to the folder where the results (plots, csv, etc.) are saved
    RESULTS_PATH = os.environ.get("PATH_RESULTS", "results/")
    os.makedirs(RESULTS_PATH, exist_ok=True)
    # Path to the folder where the tensorboard files are saved
    TENSORBOARD_PATH = os.environ.get("PATH_TENSORBOARD", "runs/")
    if TENSORBOARD_CLEAN:
        # if the folder "runs"  exists, move it to "runs_Y_M_D_H_M_S" to avoid overwriting old tensorboard files
        if os.path.exists(TENSORBOARD_PATH):
            now = datetime.datetime.now()
            os.makedirs("runs_OLD", exist_ok=True)
            # use [:-1] to remove "/" from the end of the path
            TENSORBOARD_PATH_OLD = "runs_OLD/" + TENSORBOARD_PATH[:-1] + "_" + now.strftime("%Y_%m_%d_%H_%M_%S")
            os.rename(TENSORBOARD_PATH[:-1], TENSORBOARD_PATH_OLD)
    os.makedirs(TENSORBOARD_PATH, exist_ok=True)
    if spot_tensorboard_path is not None:
        os.makedirs(spot_tensorboard_path, exist_ok=True)
        spot_writer = SummaryWriter(spot_tensorboard_path)
    else:
        spot_writer = None

    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    fun_control = {
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "DATASET_PATH": DATASET_PATH,
        "RESULTS_PATH": RESULTS_PATH,
        "TENSORBOARD_PATH": TENSORBOARD_PATH,
        "_L_in": _L_in,
        "_L_out": _L_out,
        "accelerator": accelerator,
        "counter": 0,
        "data": None,
        "data_dir": "./data",
        "data_module": None,
        "data_set": None,
        "device": device,
        "devices": devices,
        "enable_progress_bar": enable_progress_bar,
        "eval": None,
        "fun_evals": fun_evals,
        "fun_repeats": fun_repeats,
        "k_folds": 3,
        "log_level": log_level,
        "loss_function": None,
        "max_time": max_time,
        "metric_river": None,
        "metric_sklearn": None,
        "metric_torch": None,
        "metric_params": {},
        "model_dict": {},
        "noise": noise,
        "n_samples": None,
        "num_workers": num_workers,
        "optimizer": None,
        "path": None,
        "prep_model": None,
        "save_model": False,
        "seed": seed,
        "show_batch_interval": 1_000_000,
        "show_progress": show_progress,
        "shuffle": None,
        "sigma": sigma,
        "target_column": None,
        "test_seed": test_seed,
        "test_size": test_size,
        "tolerance_x": tolerance_x,
        "train": None,
        "test": None,
        "task": task,
        "spot_tensorboard_path": spot_tensorboard_path,
        "var_name": None,
        "var_type": None,
        "weights": 1.0,
        "spot_writer": spot_writer,
    }
    return fun_control


def check_and_create_dir(path):
    """Check if the path exists and create it if it does not.

    Args:
        path (str): Path to the directory.

    Returns:
        (noneType): None

    Examples:
        >>> fromspotPy.utils.init import check_and_create_dir
        >>> check_and_create_dir("data/")
    """
    if not isinstance(path, str):
        raise Exception("path must be a string")
    if not os.path.exists(path):
        os.makedirs(path)


def design_control_init():
    """Initialize design_control dictionary.

    Returns:
        design_control (dict):
            A dictionary containing the information about the design of experiments.

    """
    design_control = {}
    return design_control


def surrogate_control_init():
    """Initialize surrogate_control dictionary.

    Returns:
        surrogate_control (dict):
            A dictionary containing the information about the surrogate model.

    """
    surrogate_control = {}
    return surrogate_control
