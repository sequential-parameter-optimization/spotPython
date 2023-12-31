import os
import lightning as L
import datetime

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter


def fun_control_init(
    task="classification",
    _L_in=None,
    _L_out=None,
    enable_progress_bar=False,
    spot_tensorboard_path=None,
    TENSORBOARD_CLEAN=False,
    num_workers=0,
    device=None,
    seed=1234,
    sigma=0.0,
):
    """Initialize fun_control dictionary.
    Args:
        task (str):
            The task to perform. It can be either "classification" or "regression".
        _L_in (int):
            The number of input features.
        _L_out (int):
            The number of output features.
        enable_progress_bar (bool):
            Whether to enable the progress bar or not.
        spot_tensorboard_path (str):
            The path to the folder where the spot tensorboard files are saved.
            If None, no spot tensorboard files are saved.
        num_workers (int):
            The number of workers to use for the data loading.
        device (str):
            The device to use for the training. It can be either "cpu", "mps", or "cuda".

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
                'data': None,
                'data_dir': './data',
                'device': None,
                'enable_progress_bar': False,
                'eval': None,
                'k_folds': None,
                'loss_function': None,
                'metric_river': None,
                'metric_sklearn': None,
                'metric_torch': None,
                'metric_params': {},
                'model_dict': {},
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
        "data": None,
        "data_dir": "./data",
        "data_module": None,
        "data_set": None,
        "device": device,
        "enable_progress_bar": enable_progress_bar,
        "eval": None,
        "k_folds": 3,
        "loss_function": None,
        "metric_river": None,
        "metric_sklearn": None,
        "metric_torch": None,
        "metric_params": {},
        "model_dict": {},
        "n_samples": None,
        "num_workers": num_workers,
        "optimizer": None,
        "path": None,
        "prep_model": None,
        "save_model": False,
        "seed": seed,
        "show_batch_interval": 1_000_000,
        "shuffle": None,
        "sigma": sigma,
        "target_column": None,
        "train": None,
        "test": None,
        "task": task,
        "spot_tensorboard_path": spot_tensorboard_path,
        "var_name": [],
        "var_type": [],
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
