import os
import lightning as L

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter


def fun_control_init(
    task="classification",
    _L_in=None,
    _L_out=None,
    enable_progress_bar=False,
    tensorboard_path=None,
    num_workers=0,
    device=None,
):
    """Initialize fun_control dictionary.
    Args:
        task (str): The task to perform. It can be either "classification" or "regression".
        _L_in (int): The number of input features.
        _L_out (int): The number of output features.
        enable_progress_bar (bool): Whether to enable the progress bar or not.
        tensorboard_path (str): The path to the folder where the tensorboard files are saved.
        num_workers (int): The number of workers to use for the data loading.
        device (str): The device to use for the training. It can be either "cpu", "mps", or "cuda".
    Returns:
        fun_control (dict): A dictionary containing the information about the core model, loss function, metrics,
        and the hyperparameters.
    Example:
        >>> fun_control = fun_control_init(_L_in=64, _L_out=11, num_workers=0, device=None)
        >>> fun_control
        >>> {'CHECKPOINT_PATH': 'saved_models/',
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
                'show_batch_interval': 1000000,
                'shuffle': None,
                'target_column': None,
                'train': None,
                'test': None,
                'task': 'classification',
                'tensorboard_path': None,
                'weights': 1.0,
                'writer': None}
    """
    # Setting the seed
    L.seed_everything(42)

    if tensorboard_path is not None:
        check_and_create_dir(tensorboard_path)
        # Starting with v0.2.41, Summary Writer should be not initialized here but by Lightning
        # it is only available for compatibility reasons.
        # So, set this to None and let Lightning manage the logging.
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None

    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    # Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
    DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
    os.makedirs(DATASET_PATH, exist_ok=True)
    # Path to the folder where the results (plots, csv, etc.) are saved
    RESULTS_PATH = os.environ.get("PATH_RESULTS", "results/")
    os.makedirs(RESULTS_PATH, exist_ok=True)
    # Path to the folder where the tensorboard files are saved
    TENSORBOARD_PATH = os.environ.get("PATH_TENSORBOARD", "runs/")
    os.makedirs(TENSORBOARD_PATH, exist_ok=True)

    fun_control = {
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "DATASET_PATH": DATASET_PATH,
        "RESULTS_PATH": RESULTS_PATH,
        "TENSORBOARD_PATH": TENSORBOARD_PATH,
        "_L_in": _L_in,
        "_L_out": _L_out,
        "data": None,
        "data_dir": "./data",
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
        "show_batch_interval": 1_000_000,
        "shuffle": None,
        "target_column": None,
        "train": None,
        "test": None,
        "task": task,
        "tensorboard_path": tensorboard_path,
        "weights": 1.0,
        "writer": writer,
    }
    return fun_control


def check_and_create_dir(path):
    """Check if the path exists and create it if it does not.
    Args:
        path (str): Path to the directory.
    Returns:
        None
    """
    if not isinstance(path, str):
        raise Exception("path must be a string")
    if not os.path.exists(path):
        os.makedirs(path)
