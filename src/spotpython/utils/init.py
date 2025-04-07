import os
from typing import List, Dict, Any
import lightning as L
from scipy.optimize import differential_evolution
import numpy as np
import socket
import copy
import datetime
from dateutil.tz import tzlocal
from importlib.metadata import version, PackageNotFoundError
from spotpython.hyperparameters.values import (
    add_core_model_to_fun_control,
    get_core_model_from_name,
    get_river_core_model_from_name,
    get_metric_sklearn,
    get_prep_model,
    get_river_prep_model,
)
from spotriver.hyperdict.river_hyper_dict import RiverHyperDict


def fun_control_init(
    _L_in=None,
    _L_out=None,
    _L_cond=None,
    _torchmetric=None,
    PREFIX=None,
    TENSORBOARD_CLEAN=False,
    accelerator="auto",
    collate_fn_name=None,
    converters=None,
    core_model=None,
    core_model_name=None,
    data=None,
    data_full_train=None,
    hacky=False,  # !TODO: Documentation
    data_val=None,
    data_dir="./data",
    data_module=None,
    data_set=None,
    data_set_name=None,
    data_test=None,
    db_dict_name=None,
    design=None,
    device=None,
    devices="auto",
    enable_progress_bar=False,
    EXPERIMENT_NAME=None,
    eval=None,
    force_run=True,
    fun_evals=15,
    fun_mo2so=None,
    fun_repeats=1,
    horizon=None,
    hyperdict=None,
    infill_criterion="y",
    log_every_n_steps=50,
    log_level=50,
    lower=None,
    max_time=1,
    max_surrogate_points=30,
    metric_sklearn=None,
    metric_sklearn_name=None,
    noise=False,
    n_points=1,
    n_samples=None,
    num_sanity_val_steps=2,
    n_total=None,
    num_workers=0,
    num_nodes=1,
    ocba_delta=0,
    oml_grace_period=None,
    optimizer=None,
    penalty_NA=None,
    precision="32",
    prep_model=None,
    prep_model_name=None,
    progress_file=None,
    save_experiment=False,
    save_result=True,
    scaler=None,
    scaler_name=None,
    scenario=None,
    seed=123,
    show_config=False,
    show_models=False,
    show_progress=True,
    shuffle=None,
    shuffle_train=True,
    shuffle_val=False,
    shuffle_test=False,
    sigma=0.0,
    strategy="auto",
    surrogate=None,
    target_column=None,
    target_type=None,
    task=None,
    tensorboard_log=False,
    tensorboard_start=False,
    tensorboard_stop=False,
    test=None,
    test_seed=1234,
    test_size=0.4,
    tkagg=False,
    train=None,
    tolerance_x=0,
    upper=None,
    var_name=None,
    var_type=["num"],
    verbosity=0,
    weights=1.0,
    weight_coeff=0.0,
    weights_entry=None,
):
    """Initialize fun_control dictionary.

    Args:
        _L_in (int):
            The number of input features.
        _L_out (int):
            The number of output features.
        _L_cond (int):
            The number of conditional features.
        _torchmetric (str):
            The metric to be used by the Lighting Trainer.
            For example "mean_squared_error",
            see https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html
        accelerator (str):
            The accelerator to be used by the Lighting Trainer.
            It can be either "auto", "dp", "ddp", "ddp2", "ddp_spawn", "ddp_cpu", "gpu", "tpu".
            Default is "auto".
        collate_fn_name (str):
            The name of the collate function. Default is None.
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
        data_full_train (torch.utils.data.Dataset, optional):
            The full training dataset from which training and validation sets will be derived if data_val is None.
            Default is None.
        data_val (torch.utils.data.Dataset, optional):
            The validation dataset. Default is None. If not None, the training and validation sets are derived from
            the full training dataset (data_full_train)  and the validation dataset (data_val).
        data_module (object):
            The data module object. Default is None.
        data_set (object):
            The data set object. Default is None.
        data_set_name (str):
            The name of the data set. Default is None.
        data_test (torch.utils.data.Dataset, optional):
            The separate test dataset that will be used for testing. Default is None.
        db_dict_name (str):
            The name of the database dictionary. Default is None.
        device (str):
            The device to use for the training. It can be either "cpu", "mps", or "cuda".
        devices (str or int):
            The number of devices to use for the training/validation/testing.
            Default is 1. Can be "auto" or an integer.
        design (object):
            The experimental design object. Default is None.
        enable_progress_bar (bool):
            Whether to enable the progress bar or not.
        eval (str):
            evaluation method used in sklearn taintest.py.
            Can be "eval_test", "eval_oon_score", "train_cv" or None. Default is None.
        EXPERIMENT_NAME (str):
            The name of the experiment.
            Default is None. If None, the experiment name is generated based on the
            current date and time.
        force_run (bool):
            Whether to force the run or not. If a result file (PREFIX+"_run.pkl") exists, the run is mot
            performed and the result is loaded from the file.
            Default is False.
        fun_evals (int):
            The number of function evaluations.
        fun_mo2so (object):
            The multi-objective to single-objective transformation object. Default is None.
            If None, the first objective value is used in case of multi-objective optimization.
        fun_repeats (int):
            The number of function repeats during the optimization. this value does not affect
            the number of the repeats in the initial design (this value can be set in the
            design_control). Default is 1.
        horizon (int):
            The horizon of the time series data. Default is None.
        hyperdict (dict):
            A dictionary containing the hyperparameters. Default is None.
            For example: `spotriver.hyperdict.river_hyper_dict import RiverHyperDict`
        infill_criterion (str):
            Can be `"y"`, `"s"`, `"ei"` (negative expected improvement), or `"all"`. Default is "y".
        log_every_n_steps (int):
            Lightning: How often to log within steps. Default: 50.
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
        max_surrogate_points (int):
            The maximum number of points in the surrogate model. Default is inf.
        metric_sklearn (object):
            The metric object from the scikit-learn library. Default is None.
        metric_sklearn_name (str):
            The name of the metric object from the scikit-learn library. Default is None.
        noise (bool):
            Whether the objective function is noiy or not. Default is False.
            Affects the repeat of the function evaluations.
        n_points (int):
            The number of infill points to be generated by the surrogate in each iteration.
        num_sanity_val_steps (int):
                Lightning: Sanity check runs n validation batches before starting the training routine.
                Set it to -1 to run all batches in all validation dataloaders.
                Default: 2.
        n_samples (int):
            The number of samples in the dataset. Default is None.
        n_total (int):
            The total number of samples in the dataset. Default is None.
        num_nodes (int):
            The number of GPU nodes to use for the training/validation/testing. Default is 1.
        num_workers (int):
            The number of workers to use for the data loading. Default is 0.
        ocba_delta (int):
            The number of additional, new points (only used if noise==True) generated by
            the OCBA infill criterion. Default is 0.
        oml_grace_period (int):
            The grace period for the OML algorithm. Default is None.
        optimizer (object):
            The optimizer object used for the search on surrogate. Default is None.
        penalty_NA (float):
            The penalty for NA values. Default is None. If None, the values are ignored, e.g., the
            initial design size used for the surrogate is reduced by the number of NA values.
        precision (str):
            The precision of the data. Default is "32". Can be e.g., "16-mixed" or "16-true".
        PREFIX (str):
            The prefix of the experiment name. If the PREFIX is not None, a spotWriter
            that us an instance of a SummaryWriter(), is created. Default is "00".
        prep_model (object):
            The preprocessing model object. Used for river. Default is None.
        prep_model_name (str):
            The name of the preprocessing model. Default is None.
        progress_file (str):
            The name of the progress file. Default is None.
        save_experiment (bool):
            Whether to save the experiment before the run is started or not. Default is False.
        save_result (bool):
            Whether to save the result after the experiment is done or not. Default is False.
        scaler (object):
            The scaler object, e.g., the TorchStandard scaler from spot.utils.scaler.py.
            Default is None.
        scaler_name (str):
            The name of the scaler object. Default is None.
        scenario (str):
            The scenario to use. Default is None. Can be "river", "sklearn", or "lightning".
        seed (int):
            The seed to use for the random number generator. Default is 123.
        sigma (float):
            The standard deviation of the noise of the objective function.
        show_progress (bool):
            Whether to show the progress or not. Default is `True`.
        show_models (bool):
            Plot model each generation.
            Currently only 1-dim functions are supported. Default is `False`.
        show_config (bool):
            Whether to show the configuration or not. Default is `False`.
        shuffle (bool):
            Whether the data were shuffled or not. Default is None.
        shuffle_train (bool):
            Whether the training data were shuffled or not. Default is True.
        shuffle_val (bool):
            Whether the validation data were shuffled or not. Default is False.
        shuffle_test (bool):
            Whether the test data were shuffled or not. Default is False.
        surrogate (object):
            The surrogate model object. Default is None.
        strategy (str):
            The strategy to use. Default is "auto".
        target_column (str):
            The name of the target column. Default is None.
        target_type (str):
            The type of the target column. Default is None.
        task (str):
            The task to perform. It can be either "classification" or "regression".
            Default is None.
        TENSORBOARD_CLEAN (bool):
            Whether to clean (delete) the tensorboard folder or not. Default is False.
        tensorboard_log (bool):
            Whether to log the tensorboard or not. Starts the SummaryWriter.
            Default is False.
        tensorboard_start (bool):
            Whether to start the tensorboard or not. Default is False.
        tensorboard_stop (bool):
            Whether to stop the tensorboard or not. Default is False.
        test (object):
            The test data set for spotriver. Default is None.
        test_seed (int):
            The seed to use for the test set. Default is 1234.
        test_size (float):
            The size of the test set. Default is 0.4, i.e.,
            60% of the data is used for training and 40% for testing.
        tkagg (bool):
            Whether to use matplotlib TkAgg or not. Default is False.
        tolerance_x (float):
            tolerance for new x solutions. Minimum distance of new solutions,
            generated by `suggest_new_X`, to already existing solutions.
            If zero (which is the default), every new solution is accepted.
        train (object):
            The training data set for spotriver. Default is None.
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
            Default is 1.0.
        weight_coeff (float):
            Determines how to weight older measures. Default is 1.0. Used in the OML algorithm eval_oml.py.
            Default is 0.0.
        weights_entry (str):
            The weights entry used in the GUI. Default is None.

    Returns:
        fun_control (dict):
            A dictionary containing the information about the core model,
            loss function, metrics, and the hyperparameters.

    Examples:
        >>> from spotpython.utils.init import fun_control_init
            fun_control = fun_control_init(_L_in=64, _L_out=11, num_workers=0, device=None)
            fun_control
            {'CHECKPOINT_PATH': 'saved_models/',
                'DATASET_PATH': 'data/',
                'RESULTS_PATH': 'results/',
                'TENSORBOARD_PATH': 'runs/',
                '_L_in': 64,
                '_L_out': 11,
                '_L_cond': None,
                'accelerator': "auto",
                'core_model': None,
                'core_model_name': None,
                'data': None,
                'data_dir': './data',
                'db_dict_name': None,
                'device': None,
                'devices': "auto",
                'enable_progress_bar': False,
                'eval': None,
                'horizon': 7,
                'infill_criterion': 'y',
                'k_folds': None,
                'loss_function': None,
                'lower': None,
                'max_surrogate_points': 100,
                'metric_river': None,
                'metric_sklearn': None,
                'metric_sklearn_name': None,
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
                'prep_model_name': None,
                'save_model': False,
                'scenario': "lightning",
                'seed': 1234,
                'show_batch_interval': 1000000,
                'shuffle': None,
                'sigma': 0.0,
                'target_column': None,
                'target_type': None,
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

    if PREFIX is None:
        PREFIX = _init_prefix()

    CHECKPOINT_PATH, DATASET_PATH, RESULTS_PATH, TENSORBOARD_PATH = setup_paths(TENSORBOARD_CLEAN)
    spot_tensorboard_path = create_spot_tensorboard_path(tensorboard_log, PREFIX)

    if metric_sklearn is None and metric_sklearn_name is not None:
        metric_sklearn = get_metric_sklearn(metric_sklearn_name)

    fun_control = {
        "PREFIX": PREFIX,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "DATASET_PATH": DATASET_PATH,
        "RESULTS_PATH": RESULTS_PATH,
        "TENSORBOARD_PATH": TENSORBOARD_PATH,
        "TENSORBOARD_CLEAN": TENSORBOARD_CLEAN,
        "_L_in": _L_in,
        "_L_out": _L_out,
        "_L_cond": _L_cond,
        "_torchmetric": _torchmetric,
        "accelerator": accelerator,
        "collate_fn_name": collate_fn_name,
        "converters": converters,
        "core_model": core_model,
        "core_model_name": core_model_name,
        "counter": 0,
        "data": data,
        "data_dir": data_dir,
        "data_full_train": data_full_train,
        "hacky": hacky,
        "data_module": data_module,
        "data_set": data_set,
        "data_set_name": data_set_name,
        "data_test": data_test,
        "data_val": data_val,
        "db_dict_name": db_dict_name,
        "design": design,
        "device": device,
        "devices": devices,
        "enable_progress_bar": enable_progress_bar,
        "eval": eval,
        "force_run": force_run,
        "fun_evals": fun_evals,
        "fun_mo2so": fun_mo2so,
        "fun_repeats": fun_repeats,
        "horizon": horizon,
        "hyperdict": hyperdict,
        "infill_criterion": infill_criterion,
        "k_folds": 3,
        "log_every_n_steps": log_every_n_steps,
        "log_graph": False,
        "log_level": log_level,
        "loss_function": None,
        "lower": lower,
        "max_time": max_time,
        "max_surrogate_points": max_surrogate_points,
        "metric_river": None,
        "metric_sklearn": metric_sklearn,
        "metric_sklearn_name": metric_sklearn_name,
        "metric_torch": None,
        "metric_params": {},
        "model_dict": {},
        "noise": noise,
        "n_points": n_points,
        "n_samples": n_samples,
        "n_total": n_total,
        "num_nodes": num_nodes,
        "num_sanity_val_steps": num_sanity_val_steps,
        "num_workers": num_workers,
        "ocba_delta": ocba_delta,
        "oml_grace_period": oml_grace_period,
        "optimizer": optimizer,
        "path": None,
        "penalty_NA": penalty_NA,
        "precision": precision,
        "prep_model": prep_model,
        "prep_model_name": prep_model_name,
        "progress_file": progress_file,
        "save_experiment": save_experiment,
        "save_result": save_result,
        "save_model": False,
        "scaler": scaler,
        "scaler_name": scaler_name,
        "scenario": scenario,
        "seed": seed,
        "show_batch_interval": 1_000_000,
        "show_config": show_config,
        "show_models": show_models,
        "show_progress": show_progress,
        "shuffle": shuffle,
        "shuffle_train": shuffle_train,
        "shuffle_val": shuffle_val,
        "shuffle_test": shuffle_test,
        "sigma": sigma,
        "spot_tensorboard_path": spot_tensorboard_path,
        "strategy": strategy,
        "target_column": target_column,
        "target_type": target_type,
        "task": task,
        "tensorboard_log": tensorboard_log,
        "tensorboard_start": tensorboard_start,
        "tensorboard_stop": tensorboard_stop,
        "test": test,
        "test_seed": test_seed,
        "test_size": test_size,
        "tkagg": tkagg,
        "tolerance_x": tolerance_x,
        "train": train,
        "upper": upper,
        "var_name": var_name,
        "var_type": var_type,
        "verbosity": verbosity,
        "weights": weights,
        "weight_coeff": weight_coeff,
        "weights_entry": weights_entry,
    }
    if hyperdict is not None and core_model_name is not None:
        # check if hyperdict implements the methods get_scenario:
        if hasattr(hyperdict, "get_scenario"):
            scenario = hyperdict().get_scenario()
        else:
            scenario = None
        if fun_control["hyperdict"].__name__ == RiverHyperDict.__name__ or scenario == "river":
            coremodel, core_model_instance = get_river_core_model_from_name(core_model_name)
            if prep_model is None and prep_model_name is not None:
                prep_model = get_river_prep_model(prep_model_name)
        else:
            coremodel, core_model_instance = get_core_model_from_name(core_model_name)
            if prep_model is None and prep_model_name is not None:
                prep_model = get_prep_model(prep_model_name)
        fun_control.update({"prep_model": prep_model})
        add_core_model_to_fun_control(
            core_model=core_model_instance,
            fun_control=fun_control,
            hyper_dict=hyperdict,
            filename=None,
        )
    if hyperdict is not None and core_model is not None:
        add_core_model_to_fun_control(
            core_model=core_model,
            fun_control=fun_control,
            hyper_dict=hyperdict,
            filename=None,
        )
    return fun_control


def _init_prefix() -> str:
    """Initialize the prefix for the experiment name.
    Attempts to derive the prefix from the package version. If unsuccessful,
    defaults to '000'.

    Returns:
        str: The prefix for the experiment name.

    Examples:
        >>> from spotpython.utils.init import _init_prefix
        >>> _init_prefix()
        '00'
    """
    DEFAULT_PREFIX = "000"
    try:
        package_version = version("package_name")
    except PackageNotFoundError:
        package_version = DEFAULT_PREFIX

    return package_version


def setup_paths(tensorboard_clean) -> tuple:
    """
    Setup paths for checkpoints, datasets, results, and tensorboard files.
    This function also handles cleaning the tensorboard path if specified.

    Args:
        tensorboard_clean (bool):
            If True, move the existing tensorboard folder to a timestamped backup
            folder to avoid overwriting old tensorboard files.

    Returns:
        CHECKPOINT_PATH (str):
            The path to the folder where the pretrained models are saved.
        DATASET_PATH (str):
            The path to the folder where the datasets are/should be downloaded.
        RESULTS_PATH (str):
            The path to the folder where the results (plots, csv, etc.) are saved.
        TENSORBOARD_PATH (str):
            The path to the folder where the tensorboard files are saved.

    Examples:
        >>> from spotpython.utils.init import setup_paths
        >>> setup_paths(tensorboard_clean=True)
        ('runs/saved_models/', 'data/', 'results/', 'runs/')

    """
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
    if tensorboard_clean:
        # if the folder "runs" exists, move it to "runs_Y_M_D_H_M_S" to avoid overwriting old tensorboard files
        if os.path.exists(TENSORBOARD_PATH):
            now = datetime.datetime.now()
            os.makedirs("runs_OLD", exist_ok=True)
            # use [:-1] to remove "/" from the end of the path
            TENSORBOARD_PATH_OLD = "runs_OLD/" + TENSORBOARD_PATH[:-1] + "_" + now.strftime("%Y_%m_%d_%H_%M_%S") + "_" + "0"
            print(f"Moving TENSORBOARD_PATH: {TENSORBOARD_PATH} to TENSORBOARD_PATH_OLD: {TENSORBOARD_PATH_OLD}")
            # if TENSORBOARD_PATH_OLD already exists, change the name increasing the number at the end
            while os.path.exists(TENSORBOARD_PATH_OLD):
                TENSORBOARD_PATH_OLD = copy.deepcopy(TENSORBOARD_PATH_OLD[:-1] + str(int(TENSORBOARD_PATH_OLD[-1]) + 1))
            os.rename(TENSORBOARD_PATH[:-1], TENSORBOARD_PATH_OLD)

    os.makedirs(TENSORBOARD_PATH, exist_ok=True)

    # Ensure the figures folder exists
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    return CHECKPOINT_PATH, DATASET_PATH, RESULTS_PATH, TENSORBOARD_PATH


def create_spot_tensorboard_path(tensorboard_log, prefix) -> str:
    """Creates the spot_tensorboard_path and returns it.

    Args:
        tensorboard_log (bool):
            If True, the path to the folder where the tensorboard files are saved is created.
        prefix (str):
            The prefix for the experiment name.

    Returns:
        spot_tensorboard_path (str):
            The path to the folder where the tensorboard files are saved.
    """
    if tensorboard_log:
        experiment_name = get_experiment_name(prefix=prefix)
        spot_tensorboard_path = get_spot_tensorboard_path(experiment_name)
        os.makedirs(spot_tensorboard_path, exist_ok=True)
        print(f"Created spot_tensorboard_path: {spot_tensorboard_path} for SummaryWriter()")
    else:
        spot_tensorboard_path = None
    return spot_tensorboard_path


def X_reshape(X) -> np.array:
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


def check_and_create_dir(path) -> None:
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
    method="regression",
    model_optimizer=differential_evolution,
    model_fun_evals=10000,
    min_theta=-3.0,
    max_theta=2.0,
    n_theta="anisotropic",
    p_val=2.0,
    n_p=1,
    optim_p=False,
    min_Lambda=1e-9,
    max_Lambda=1,
    seed=124,
    theta_init_zero=False,
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
        method (str):
            The method to be used for the surrogate model. Default is "regression".
            Can be one of ["regression", "interpolation", "reinterpolation"].
            Note: Will also be set in the Spot class, if None.
        n_theta (int):
            The number of theta values. If larger than 1 or set to the string "anisotropic",
            then the k theta values are used, where k is the problem dimension.
            This is handled in spot.py. Default is "anisotropic".
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
            Whether to initialize theta with zero or not. If False, theta is
            set to n/(100 * k). Default is False.
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
            * `method`: If the surrogate model dictionary is passed to the Spot class,
              and the `method` value is `None`, then the method value is set in the
              Spot class based on the value of `method` in the Spot class fun_control dictionary.
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
        "method": method,
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
) -> dict:
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
        >>> from spotpython.utils.init import get_experiment_name
        >>> get_experiment_name(prefix="00")
        00_ubuntu_2021-08-31_14-30-00
    """
    start_time = datetime.datetime.now(tzlocal())
    HOSTNAME = socket.gethostname().split(".")[0]
    experiment_name = prefix + "_" + HOSTNAME + "_" + str(start_time).split(".", 1)[0].replace(" ", "_")
    experiment_name = experiment_name.replace(":", "-")
    return experiment_name


def get_spot_tensorboard_path(experiment_name) -> str:
    """Get the path to the spot tensorboard files.

    Args:
        experiment_name (str): The name of the experiment.

    Returns:
        spot_tensorboard_path (str): The path to the folder where the spot tensorboard files are saved.

    Examples:
        >>> from spotpython.utils.init import get_spot_tensorboard_path
        >>> get_spot_tensorboard_path("00_ubuntu_2021-08-31_14-30-00")
        runs/spot_logs/00_ubuntu_2021-08-31_14-30-00

    """
    spot_tensorboard_path = os.environ.get("PATH_TENSORBOARD", "runs/spot_logs/")
    spot_tensorboard_path = os.path.join(spot_tensorboard_path, experiment_name)
    return spot_tensorboard_path


def get_tensorboard_path(fun_control) -> str:
    """Get the path to the tensorboard files.

    Args:
        fun_control (dict): The function control dictionary.

    Returns:
        tensorboard_path (str): The path to the folder where the tensorboard files are saved.

    Examples:
        >>> from spotpython.utils.init import get_tensorboard_path
        >>> get_tensorboard_path(fun_control)
        runs/
    """
    return fun_control["TENSORBOARD_PATH"]


def get_feature_names(fun_control: Dict[str, Any]) -> List[str]:
    """
    Get the feature names from the fun_control dictionary.

    Args:
        fun_control (dict): The function control dictionary. Must contain a "data_set" key.

    Returns:
        List[str]: List of feature names.

    Raises:
        ValueError: If "data_set" is not in fun_control.
        ValueError: If "data_set" is None.

    Examples:
        >>> from spotpython.utils.init import get_feature_names
            get_feature_names(fun_control)
    """
    data_set = fun_control.get("data_set")

    if data_set is None:
        raise ValueError("'data_set' key not found or is None in 'fun_control'")

    return data_set.names
