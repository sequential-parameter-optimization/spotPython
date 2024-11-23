import lightning as L
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.utils.eda import generate_config_id
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import os


def generate_config_id_with_timestamp(config: dict, timestamp: bool) -> str:
    """
    Generates a configuration ID based on the given config and timestamp flag.

    Args:
        config (dict): The configuration parameters.
        timestamp (bool): Indicates whether to include a timestamp in the config ID.

    Returns:
        str: The generated configuration ID.
    """
    if timestamp:
        # config id is unique. Since the model is not loaded from a checkpoint,
        # the config id is generated here with a timestamp.
        config_id = generate_config_id(config, timestamp=True)
    else:
        # config id is not time-dependent and therefore unique,
        # so that the model can be loaded from a checkpoint,
        # the config id is generated here without a timestamp.
        config_id = generate_config_id(config, timestamp=False) + "_TRAIN"
    return config_id


def build_model_instance(config: dict, fun_control: dict) -> L.LightningModule:
    """
    Builds the core model using the configuration and function control parameters.

    Args:
        config (dict): Model configuration parameters.
        fun_control (dict): Function control parameters.

    Returns:
        The constructed core model.
    """
    _L_in = fun_control["_L_in"]
    _L_out = fun_control["_L_out"]
    _L_cond = fun_control["_L_cond"]
    _torchmetric = fun_control["_torchmetric"]
    return fun_control["core_model"](**config, _L_in=_L_in, _L_out=_L_out, _L_cond=_L_cond, _torchmetric=_torchmetric)


def train_model(config: dict, fun_control: dict, timestamp: bool = True) -> float:
    """
    Trains a model using the given configuration and function control parameters.

    Args:
        config (dict):
            A dictionary containing the configuration parameters for the model.
        fun_control (dict):
            A dictionary containing the function control parameters.
        timestamp (bool):
            A boolean value indicating whether to include a timestamp in the config id. Default is True.
            If False, the string "_TRAIN" is appended to the config id.

    Returns:
        float: The validation loss of the trained model.

    Examples:
        >>> from math import inf
            import numpy as np
            from spotpython.data.diabetes import Diabetes
            from spotpython.hyperdict.light_hyper_dict import LightHyperDict
            from spotpython.utils.init import fun_control_init
            from spotpython.utils.eda import gen_design_table
            from spotpython.hyperparameters.values import get_default_hyperparameters_as_array
            from spotpython.hyperparameters.values import assign_values, generate_one_config_from_var_dict, get_var_name
            from spotpython.light.trainmodel import train_model
            import pprint
            PREFIX="000"
            data_set = Diabetes()
            fun_control = fun_control_init(
                PREFIX=PREFIX,
                save_experiment=True,
                fun_evals=inf,
                max_time=1,
                data_set = data_set,
                core_model_name="light.regression.NNLinearRegressor",
                hyperdict=LightHyperDict,
                _L_in=10,
                _L_out=1,
                TENSORBOARD_CLEAN=True,
                tensorboard_log=True,
                seed=42,)
            print(gen_design_table(fun_control))
            X = get_default_hyperparameters_as_array(fun_control)
            # set epochs to 2^8:
            # X[0, 1] = 8
            # set patience to 2^10:
            # X[0, 7] = 10
            print(f"X: {X}")
            # combine X and X to a np.array with shape (2, n_hyperparams)
            # so that two values are returned
            X = np.vstack((X, X))
            var_dict = assign_values(X, get_var_name(fun_control))
            for config in generate_one_config_from_var_dict(var_dict, fun_control):
                pprint.pprint(config)
                y = train_model(config, fun_control)
    """
    if fun_control["data_module"] is None:
        dm = LightDataModule(
            dataset=fun_control["data_set"],
            data_full_train=fun_control["data_full_train"],
            data_test=fun_control["data_test"],
            batch_size=config["batch_size"],
            num_workers=fun_control["num_workers"],
            test_size=fun_control["test_size"],
            test_seed=fun_control["test_seed"],
            scaler=fun_control["scaler"],
            verbosity=fun_control["verbosity"],
        )
    else:
        dm = fun_control["data_module"]
    model = build_model_instance(config, fun_control)
    # TODO: Check if this is necessary or if this is handled by the trainer
    # dm.setup()
    # print(f"train_model(): Test set size: {len(dm.data_test)}")
    # print(f"train_model(): Train set size: {len(dm.data_train)}")
    # print(f"train_model(): Batch size: {config['batch_size']}")

    # Callbacks
    #
    # EarlyStopping:
    # Stop training when a monitored quantity has stopped improving.
    # The EarlyStopping callback runs at the end of every validation epoch by default.
    # However, the frequency of validation can be modified by setting various parameters
    # in the Trainer, for example check_val_every_n_epoch and val_check_interval.
    # It must be noted that the patience parameter counts the number of validation checks
    # with no improvement, and not the number of training epochs.
    # Therefore, with parameters check_val_every_n_epoch=10 and patience=3,
    # the trainer will perform at least 40 training epochs before being stopped.
    # Args:
    # - monitor:
    #   Quantity to be monitored. Default: 'val_loss'.
    # - patience:
    #   Number of validation checks with no improvement after which training will be stopped.
    #   In spotpython, this is a hyperparameter.
    # - mode (str):
    #   one of {min, max}. If save_top_k != 0, the decision to overwrite the current save file
    #   is made based on either the maximization or the minimization of the monitored quantity.
    #   For 'val_acc', this should be 'max', for 'val_loss' this should be 'min', etc.
    # - strict:
    #   Set to False.
    # - verbose:
    #   If True, prints a message to the logger.
    #
    # ModelCheckpoint:
    # Save the model periodically by monitoring a quantity.
    # Every metric logged with log() or log_dict() is a candidate for the monitor key.
    # spotpython uses ModelCheckpoint if timestamp is set to False. In this case, the
    # config_id has no timestamp and ends with the unique string "_TRAIN". This
    # enables loading the model from a checkpoint, because the config_id is unique.
    # Args:
    # - dirpath:
    #   Path to the directory where the checkpoints will be saved.
    # - monitor (str):
    #   Quantity to monitor.
    #   By default it is None which saves a checkpoint only for the last epoch.
    # - verbose (bool):
    #   If True, prints a message to the logger.
    # - save_last (Union[bool, Literal['link'], None]):
    #   When True, saves a last.ckpt copy whenever a checkpoint file gets saved.
    #   Can be set to 'link' on a local filesystem to create a symbolic link.
    #   This allows accessing the latest checkpoint in a deterministic manner.
    #   Default: None.
    config_id = generate_config_id_with_timestamp(config=config, timestamp=timestamp)
    callbacks = [EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min", strict=False, verbose=False)]
    if not timestamp:
        # add ModelCheckpoint only if timestamp is False
        dirpath = os.path.join(fun_control["CHECKPOINT_PATH"], config_id)
        callbacks.append(ModelCheckpoint(dirpath=dirpath, monitor=None, verbose=False, save_last=True))  # Save the last checkpoint

    # Tensorboard logger. The tensorboard is passed to the trainer.
    # See: https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html
    # It uses the following arguments:
    # Args:
    # - save_dir:
    #   Where to save logs. Can be specified via fun_control["TENSORBOARD_PATH"]
    # - name:
    #   Experiment name. Defaults to 'default'.
    #   If it is the empty string then no per-experiment subdirectory is used.
    #   Changed in spotpython 0.17.2 to the empty string.
    # - version:
    #   Experiment version. If version is not specified the logger inspects the save directory
    #   for existing versions, then automatically assigns the next available version.
    #   If it is a string then it is used as the run-specific subdirectory name,
    #   otherwise 'version_${version}' is used. spotpython uses the config_id as version.
    # - log_graph (bool):
    #   Adds the computational graph to tensorboard.
    #   This requires that the user has defined the self.example_input_array
    #   attribute in their model. Set in spotpython to fun_control["log_graph"].
    # - default_hp_metric (bool):
    #   Enables a placeholder metric with key hp_metric when log_hyperparams is called
    #   without a metric (otherwise calls to log_hyperparams without a metric are ignored).
    #   spotpython sets this to True.

    # Init trainer. See: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer
    # Args used by spotpython (there are more):
    # - default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
    #   Default: os.getcwd(). Can be remote file paths such as s3://mybucket/path or ‘hdfs://path/’
    # - max_epochs: Stop training once this number of epochs is reached.
    #   Disabled by default (None).
    #   If both max_epochs and max_steps are not specified, defaults to max_epochs = 1000.
    #   To enable infinite training, set max_epochs = -1.
    # - accelerator: Supports passing different accelerator types
    #   (“cpu”, “gpu”, “tpu”, “hpu”, “mps”, “auto”) as well as custom accelerator instances.
    # - devices: The devices to use. Can be set to a positive number (int or str),
    #   a sequence of device indices (list or str), the value -1 to indicate all available devices
    #   should be used, or "auto" for automatic selection based on the chosen accelerator.
    #   Default: "auto".
    # - strategy: Supports different training strategies with aliases as well custom strategies.
    #   Default: "auto".
    # - num_nodes: Number of GPU nodes for distributed training. Default: 1.
    # - precision: Double precision (64, ‘64’ or ‘64-true’), full precision (32, ‘32’ or ‘32-true’),
    #   16bit mixed precision (16, ‘16’, ‘16-mixed’) or bfloat16 mixed precision (‘bf16’, ‘bf16-mixed’).
    #   Can be used on CPU, GPU, TPUs, or HPUs. Default: '32-true'.
    # - logger: Logger (or iterable collection of loggers) for experiment tracking.
    #   A True value uses the default TensorBoardLogger if it is installed, otherwise CSVLogger.
    #   False will disable logging. If multiple loggers are provided, local files (checkpoints,
    #   profiler traces, etc.) are saved in the log_dir of the first logger. Default: True.
    # - callbacks: List of callbacks to enable during training.Default: None.
    # - enable_progress_bar: If True, enables the progress bar.
    #   Whether to enable to progress bar by default. Default: True.
    # - num_sanity_val_steps:
    #   Sanity check runs n validation batches before starting the training routine.
    #   Set it to -1 to run all batches in all validation dataloaders. Default: 2.
    # - log_every_n_steps:
    #   How often to log within steps. Default: 50.
    # - gradient_clip_val:
    #   The value at which to clip gradients. Passing gradient_clip_val=None
    #   disables gradient clipping. If using Automatic Mixed Precision (AMP),
    #   the gradients will be unscaled before. Default: None.
    # - gradient_clip_algorithm (str):
    #   The gradient clipping algorithm to use.
    #   Pass gradient_clip_algorithm="value" to clip by value,
    #   and gradient_clip_algorithm="norm" to clip by norm.
    #   By default it will be set to "norm".

    enable_progress_bar = fun_control["enable_progress_bar"] or False
    trainer = L.Trainer(
        # Where to save models
        default_root_dir=os.path.join(fun_control["CHECKPOINT_PATH"], config_id),
        max_epochs=model.hparams.epochs,
        accelerator=fun_control["accelerator"],
        devices=fun_control["devices"],
        strategy=fun_control["strategy"],
        num_nodes=fun_control["num_nodes"],
        precision=fun_control["precision"],
        logger=TensorBoardLogger(save_dir=fun_control["TENSORBOARD_PATH"], version=config_id, default_hp_metric=True, log_graph=fun_control["log_graph"], name=""),
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        num_sanity_val_steps=fun_control["num_sanity_val_steps"],
        log_every_n_steps=fun_control["log_every_n_steps"],
        gradient_clip_val=None,
        gradient_clip_algorithm="norm",
    )

    # Fit the model
    # Args:
    # - model: Model to fit
    # - datamodule: A LightningDataModule that defines the train_dataloader
    #   hook. Pass the datamodule as arg to trainer.fit to override model hooks # :)
    # - ckpt_path: Path/URL of the checkpoint from which training is resumed.
    #   Could also be one of two special keywords "last" and "hpc".
    #   If there is no checkpoint file at the path, an exception is raised.
    trainer.fit(model=model, datamodule=dm, ckpt_path=None)

    # Test best model on validation and test set

    verbose = fun_control["verbosity"] > 0

    # Validate the model
    # Perform one evaluation epoch over the validation set.
    # Args:
    # - model: The model to validate.
    # - datamodule: A LightningDataModule that defines the val_dataloader hook.
    # - verbose: If True, prints the validation results.
    # - ckpt_path: Path to a specific checkpoint to load for validation.
    #   Either "best", "last", "hpc" or path to the checkpoint you wish to validate.
    #   If None and the model instance was passed, use the current weights.
    #   Otherwise, the best model checkpoint from the previous trainer.fit call will
    #   be loaded if a checkpoint callback is configured.
    # Returns:
    # - List of dictionaries with metrics logged during the validation phase,
    #   e.g., in model- or callback hooks like validation_step() etc.
    #   The length of the list corresponds to the number of validation dataloaders used.
    result = trainer.validate(model=model, datamodule=dm, ckpt_path=None, verbose=verbose)

    # unlist the result (from a list of one dict)
    result = result[0]
    print(f"train_model result: {result}")
    return result["val_loss"]
