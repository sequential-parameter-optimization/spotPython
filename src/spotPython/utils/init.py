import os
import lightning as L
from scipy.optimize import differential_evolution
import numpy as np
import socket
import datetime
from dateutil.tz import tzlocal
from torch.utils.tensorboard import SummaryWriter


def fun_control_init(
    _L_in=None,
    _L_out=None,
    _torchmetric=None,
    PREFIX=None,
    TENSORBOARD_CLEAN=False,
    SUMMARY_WRITER=True,
    accelerator="auto",
    converters=None,
    core_model=None,
    core_model_name=None,
    data=None,
    data_dir="./data",
    data_module=None,
    data_set=None,
    data_set_name=None,
    design=None,
    device=None,
    devices=1,
    enable_progress_bar=False,
    EXPERIMENT_NAME=None,
    fun_evals=15,
    fun_repeats=1,
    horizon=None,
    infill_criterion="y",
    log_level=50,
    lower=None,
    max_time=1,
    metric_sklearn=None,
    noise=False,
    n_points=1,
    n_samples=None,
    n_total=None,
    num_workers=0,
    ocba_delta=0,
    oml_grace_period=None,
    optimizer=None,
    prep_model=None,
    seed=123,
    show_models=False,
    show_progress=True,
    sigma=0.0,
    surrogate=None,
    target_column=None,
    task=None,
    test=None,
    test_seed=1234,
    test_size=0.4,
    train=None,
    tolerance_x=0,
    upper=None,
    var_name=None,
    var_type=["num"],
    verbosity=0,
    weights=1.0,
    weight_coeff=0.0,
):
    """Initialize fun_control dictionary.

    Args:
        _L_in (int):
            The number of input features.
        _L_out (int):
            The number of output features.
        _torchmetric (str):
            The metric to be used by the Lighting Trainer.
            For example "mean_squared_error",
            see https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html
        accelerator (str):
            The accelerator to be used by the Lighting Trainer.
            It can be either "auto", "dp", "ddp", "ddp2", "ddp_spawn", "ddp_cpu", "gpu", "tpu".
            Default is "auto".
        converters (dict):
            A dictionary containing the converters. Default is None.
        core_model (object):
            The core model object. Default is None.
        core_model_name (str):
            The name of the core model. Default is None.
        data (object):
            The data object. Default is None.
        data_dir (str):
            The directory to save the data. Default is "./data".
        data_module (object):
            The data module object. Default is None.
        data_set (object):
            The data set object. Default is None.
        data_set_name (str):
            The name of the data set. Default is None.
        device (str):
            The device to use for the training. It can be either "cpu", "mps", or "cuda".
        devices (str or int):
            The number of devices to use for the training/validation/testing.
            Default is 1. Can be "auto" or an integer.
        design (object):
            The experimental design object. Default is None.
        enable_progress_bar (bool):
            Whether to enable the progress bar or not.
        EXPERIMENT_NAME (str):
            The name of the experiment. If EXPERIMENT_NAME is not None,
            the spot_tensorboard_path is compiled and
            a spot_writer is initialized as a SummaryWriter(spot_tensorboard_path). Default is None.
        fun_evals (int):
            The number of function evaluations.
        fun_repeats (int):
            The number of function repeats during the optimization. this value does not affect
            the number of the repeats in the initial design (this value can be set in the
            design_control). Default is 1.
        horizon (int):
            The horizon of the time series data. Default is None.
        infill_criterion (str):
            Can be `"y"`, `"s"`, `"ei"` (negative expected improvement), or `"all"`. Default is "y".
        log_level (int):
            log level with the following settings:
            `NOTSET` (`0`),
            `DEBUG` (`10`: Detailed information, typically of interest only when diagnosing problems.),
            `INFO` (`20`: Confirmation that things are working as expected.),
            `WARNING` (`30`: An indication that something unexpected happened, or indicative of some problem in the near
                future (e.g. ‘disk space low’). The software is still working as expected.),
            `ERROR` (`40`: Due to a more serious problem, the software has not been able to perform some function.), and
            `CRITICAL` (`50`: A serious error, indicating that the program itself may be unable to continue running.)
        lower (np.array):
            lower bound
        max_time (int):
            The maximum time in minutes.
        metric_sklearn (object):
            The metric object from the scikit-learn library. Default is None.
        noise (bool):
            Whether the objective function is noiy or not. Default is False.
            Affects the repeat of the function evaluations.
        n_points (int):
            The number of infill points to be generated by the surrogate in each iteration.
        n_samples (int):
            The number of samples in the dataset. Default is None.
        n_total (int):
            The total number of samples in the dataset. Default is None.
        num_workers (int):
            The number of workers to use for the data loading. Default is 0.
        ocba_delta (int):
            The number of additional, new points (only used if noise==True) generated by
            the OCBA infill criterion. Default is 0.
        oml_grace_period (int):
            The grace period for the OML algorithm. Default is None.
        optimizer (object):
            The optimizer object used for the search on surrogate. Default is None.
        PREFIX (str):
            The prefix of the experiment name. If the PREFIX is not None, a spotWriter
            that us an instance of a SummaryWriter(), is created. Default is None.
        prep_model (object):
            The preprocessing model object. Used for river. Default is None.
        seed (int):
            The seed to use for the random number generator. Default is 123.
        sigma (float):
            The standard deviation of the noise of the objective function.
        show_progress (bool):
            Whether to show the progress or not. Default is `True`.
        show_models (bool):
            Plot model each generation.
            Currently only 1-dim functions are supported. Default is `False`.
        surrogate (object):
            The surrogate model object. Default is None.
        target_column (str):
            The name of the target column. Default is None.
        task (str):
            The task to perform. It can be either "classification" or "regression".
            Default is None.
        TENSORBOARD_CLEAN (bool):
            Whether to clean (delete) the tensorboard folder or not. Default is False.
        test (object):
            The test data set for spotRiver. Default is None.
        test_seed (int):
            The seed to use for the test set. Default is 1234.
        test_size (float):
            The size of the test set. Default is 0.4, i.e.,
            60% of the data is used for training and 40% for testing.
        tolerance_x (float):
            tolerance for new x solutions. Minimum distance of new solutions,
            generated by `suggest_new_X`, to already existing solutions.
            If zero (which is the default), every new solution is accepted.
        train (object):
            The training data set for spotRiver. Default is None.
        upper (np.array):
            upper bound
        var_name (list):
            A list containing the name of the variables, e.g., ["x1", "x2"]. Default is None.
        var_type (List[str]):
            list of type information, can be either "int", "num" or "factor".
            Default is ["num"].
        verbosity (int):
            The verbosity level. Determines print output to console. Higher values
            result in more output. Default is 0.
        weights (float):
            The weight coefficient of the objective function. Positive values mean minimization.
            If set to -1, scores that are better when maximized will be minimized, e.g, accuracy.
            Can be an array, so that different weights can be used for different (multiple) objectives.
        weight_coeff (float):
            Determines how to weight older measures. Default is 1.0. Used in the OML algorithm eval_oml.py.

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
                'core_model': None,
                'core_model_name': None,
                'data': None,
                'data_dir': './data',
                'device': None,
                'devices': "auto",
                'enable_progress_bar': False,
                'eval': None,
                'horizon': 7,
                'infill_criterion': 'y',
                'k_folds': None,
                'loss_function': None,
                'lower': None,
                'metric_river': None,
                'metric_sklearn': None,
                'metric_torch': None,
                'metric_params': {},
                'model_dict': {},
                'noise': False,
                'n_points': 1,
                'n_samples': None,
                'num_workers': 0,
                'ocba_delta': 0,
                'oml_grace_period': None,
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
                'upper': None,
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
            print(f"Moving TENSORBOARD_PATH: {TENSORBOARD_PATH} to TENSORBOARD_PATH_OLD: {TENSORBOARD_PATH_OLD}")
            os.rename(TENSORBOARD_PATH[:-1], TENSORBOARD_PATH_OLD)
    os.makedirs(TENSORBOARD_PATH, exist_ok=True)
    if PREFIX is not None and SUMMARY_WRITER:
        experiment_name = get_experiment_name(prefix=PREFIX)
        spot_tensorboard_path = get_spot_tensorboard_path(experiment_name)
        os.makedirs(spot_tensorboard_path, exist_ok=True)
        print(f"Created spot_tensorboard_path: {spot_tensorboard_path} for SummaryWriter()")
        spot_writer = SummaryWriter(log_dir=spot_tensorboard_path)
    else:
        spot_writer = None
        spot_tensorboard_path = None

    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    fun_control = {
        "PREFIX": PREFIX,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "DATASET_PATH": DATASET_PATH,
        "RESULTS_PATH": RESULTS_PATH,
        "TENSORBOARD_PATH": TENSORBOARD_PATH,
        "_L_in": _L_in,
        "_L_out": _L_out,
        "_torchmetric": _torchmetric,
        "accelerator": accelerator,
        "converters": converters,
        "core_model": core_model,
        "core_model_name": core_model_name,
        "counter": 0,
        "data": data,
        "data_dir": data_dir,
        "data_module": data_module,
        "data_set": data_set,
        "data_set_name": data_set_name,
        "design": design,
        "device": device,
        "devices": devices,
        "enable_progress_bar": enable_progress_bar,
        "eval": None,
        "fun_evals": fun_evals,
        "fun_repeats": fun_repeats,
        "horizon": horizon,
        "infill_criterion": infill_criterion,
        "k_folds": 3,
        "log_graph": False,
        "log_level": log_level,
        "loss_function": None,
        "lower": lower,
        "max_time": max_time,
        "metric_river": None,
        "metric_sklearn": metric_sklearn,
        "metric_torch": None,
        "metric_params": {},
        "model_dict": {},
        "noise": noise,
        "n_points": n_points,
        "n_samples": n_samples,
        "n_total": n_total,
        "num_workers": num_workers,
        "ocba_delta": ocba_delta,
        "oml_grace_period": oml_grace_period,
        "optimizer": optimizer,
        "path": None,
        "prep_model": prep_model,
        "save_model": False,
        "seed": seed,
        "show_batch_interval": 1_000_000,
        "show_models": show_models,
        "show_progress": show_progress,
        "shuffle": None,
        "sigma": sigma,
        "spot_tensorboard_path": spot_tensorboard_path,
        "spot_writer": spot_writer,
        "target_column": target_column,
        "task": task,
        "test": test,
        "test_seed": test_seed,
        "test_size": test_size,
        "tolerance_x": tolerance_x,
        "train": train,
        "upper": upper,
        "var_name": var_name,
        "var_type": var_type,
        "verbosity": verbosity,
        "weights": weights,
        "weight_coeff": weight_coeff,
    }
    # lower = X_reshape(lower)
    # fun_control.update({"lower": lower})
    # upper = X_reshape(upper)
    # fun_control.update({"upper": upper})
    return fun_control


def X_reshape(X):
    """Reshape X to 2D array.

    Args:
        X (np.array):
            The input array.

    Returns:
        X (np.array):
            The reshaped input array.

    Examples:
        >>> from spotPy.utils.init import X_reshape
        >>> X = np.array([1,2,3])
        >>> X_reshape(X)
        array([[1, 2, 3]])
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    X = np.atleast_2d(X)
    return X


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


def design_control_init(init_size=10, repeats=1) -> dict:
    """Initialize design_control dictionary.

    Args:
        init_size (int): The initial size of the experimental design.
        repeats (int): The number of repeats of the design.

    Returns:
        design_control (dict):
            A dictionary containing the information about the design of experiments.

    """
    design_control = {"init_size": init_size, "repeats": repeats}
    return design_control


def surrogate_control_init(
    log_level: int = 50,
    noise=False,
    model_optimizer=differential_evolution,
    model_fun_evals=10000,
    min_theta=-3.0,
    max_theta=2.0,
    n_theta=1,
    p_val=2.0,
    n_p=1,
    optim_p=False,
    min_Lambda=1e-9,
    max_Lambda=1,
    seed=124,
    theta_init_zero=True,
    var_type=None,
    metric_factorial="canberra",
) -> dict:
    """Initialize surrogate_control dictionary.

    Args:
        model_optimizer (object):
            The optimizer object used for the search on surrogate.
            Default is differential_evolution.
        model_fun_evals (int):
            The number of function evaluations. This will be used for the
            optimization of the surrogate model. Default is 1000.
        min_theta (float):
            The minimum value of theta. Note that the base10-logarithm is used.
             Default is -3.
        max_theta (float): The maximum value of theta. Note that the base10-logarithm is used.
            Default is 3.
        noise (bool):
            Whether the objective function is noisy or not. If Kriging, then a nugget is added.
            Default is False. Note: Will be set in the Spot class.
        n_theta (int):
            The number of theta values. If larger than 1, then the k theta values are
            used, where k is the problem dimension. Default is 1.
        p_val (float):
                p value. Used as an initial value if optim_p = True. Otherwise as a constant. Defaults to 2.0.
        n_p (int):
            The number of p values. Number of p values to be used. Default is 1.
        optim_p (bool):
            Whether to optimize p or not.
        min_Lambda (float):
            The minimum value of lambda. Default is 1e-9.
        max_Lambda (float):
            The maximum value of lambda. Default is 1.
        seed (int):
            The seed to use for the random number generator.
        theta_init_zero (bool):
            Whether to initialize theta with zero or not. Default is True.
        var_type (list):
            A list containing the type of the variables. Default is None.
            Note: Will be set in the Spot class.
        metric_factorial (str):
            The metric to be used for the factorial design. Default is "canberra".

    Returns:
        surrogate_control (dict):
            A dictionary containing the information about the surrogate model.

    Note:
        * The surrogate_control dictionary is used in the Spot class. The following values
          are updated in the Spot class if they are None in the surrogate_control dictionary:
            * `noise`: If the surrogate model dictionary is passed to the Spot class,
              and the `noise` value is `None`, then the noise value is set in the
              Spot class based on the value of `noise` in the Spot class fun_control dictionary.
            * `var_type`: The `var_type` value is set in the Spot class based on the value
               of `var_type` in the Spot class fun_control dictionary and the dimension of the problem.
               If the Kriging model is used as a surrogate in the Spot class, the setting from
                surrogate_control_init() is overwritten.
            * `n_theta`: If self.surrogate_control["n_theta"] > 1,
               use k theta values, where k is the problem dimension specified in the Spot class.
               The problem dimension is set in the Spot class based on the
               length of the lower bounds.
        * This value `model_fun_evals` will used for the optimization of the surrogate model, e.g., theta values.
          Differential evaluation uses `maxiter = 1000` and sets the number of function evaluations to
          (maxiter + 1) * popsize * N, which results in 1000 * 15 * k,
          because the default popsize is 15 and N is the number of parameters. This is already sufficient
          for many situations. For example, for k=2 these are 30 000 iterations.
          Therefore we set this value to 1000.

    """
    surrogate_control = {
        "log_level": log_level,
        "noise": noise,
        "model_optimizer": model_optimizer,
        "model_fun_evals": model_fun_evals,
        "min_theta": min_theta,
        "max_theta": max_theta,
        "n_theta": n_theta,
        "p_val": p_val,
        "n_p": n_p,
        "optim_p": optim_p,
        "min_Lambda": min_Lambda,
        "max_Lambda": max_Lambda,
        "seed": seed,
        "theta_init_zero": theta_init_zero,
        "var_type": var_type,
        "metric_factorial": metric_factorial,
    }
    return surrogate_control


def optimizer_control_init(
    max_iter=1000,
    seed=125,
):
    """Initialize optimizer_control dictionary.

    Args:
        max_iter (int):
            The maximum number of iterations. This will be used for the
            optimization of the surrogate model. Default is 1000.
        seed (int):
            The seed to use for the random number generator.
            Default is 125.

    Notes:
        * Differential evaluation uses `maxiter = 1000` and sets the number of function evaluations to
          (maxiter + 1) * popsize * N, which results in 1000 * 15 * k,
          because the default popsize is 15 and N is the number of parameters. This is already sufficient
          for many situations. For example, for k=2 these are 30 000 iterations.
          Therefore we set this value to 1000.
        * This value will be passed to the surrogate model in the `Spot` class.

    Returns:
        optimizer_control (dict):
            A dictionary containing the information about the optimizer.

    """
    optimizer_control = {"max_iter": max_iter, "seed": seed}
    return optimizer_control


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
    start_time = datetime.datetime.now(tzlocal())
    HOSTNAME = socket.gethostname().split(".")[0]
    experiment_name = prefix + "_" + HOSTNAME + "_" + str(start_time).split(".", 1)[0].replace(" ", "_")
    experiment_name = experiment_name.replace(":", "-")
    return experiment_name


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
