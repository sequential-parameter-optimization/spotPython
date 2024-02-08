import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from river import preprocessing
from river.forest import AMFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.linear_model import LogisticRegression
from math import inf
import pylab
from spotRiver.data.river_hyper_dict import RiverHyperDict
from spotRiver.utils.data_conversion import convert_to_df
from spotRiver.evaluation.eval_bml import eval_oml_horizon
from spotRiver.fun.hyperriver import HyperRiver
from spotRiver.data.selector import data_selector
from spotRiver.evaluation.eval_bml import plot_bml_oml_horizon_metrics
from spotPython.plot.validation import plot_roc_from_dataframes
from spotPython.plot.validation import plot_confusion_matrix
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.utils.init import fun_control_init
from spotPython.hyperparameters.values import set_control_key_value

# from spotPython.hyperparameters.values import modify_hyper_parameter_levels
from spotPython.hyperparameters.values import get_one_core_model_from_X
from spotPython.hyperparameters.values import get_default_hyperparameters_as_array
from spotPython.spot import spot
from spotPython.utils.tensorboard import start_tensorboard
from spotPython.hyperparameters.values import set_control_hyperparameter_value
from spotPython.utils.init import design_control_init, surrogate_control_init
# Package Loading
import numpy as np
from math import inf

from spotPython.utils.device import getDevice
from spotPython.utils.init import fun_control_init
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.hyperparameters.values import set_control_hyperparameter_value
from spotPython.utils.eda import gen_design_table
from spotPython.utils.init import design_control_init, surrogate_control_init
from spotPython.fun.hyperlight import HyperLight
from spotPython.spot import spot
from spotPython.utils.eda import gen_design_table
from spotPython.hyperparameters.values import get_tuned_architecture
from spotPython.light.loadmodel import load_light_from_checkpoint
from spotPython.light.cvmodel import cv_model
from torch.utils.data import DataLoader
from spotPython.utils.init import fun_control_init
from spotPython.hyperparameters.values import set_control_key_value
from spotPython.data.diabetes import Diabetes
from spotPython.light.regression.netlightregression import NetLightRegression
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.hyperparameters.values import (
        get_default_hyperparameters_as_array, get_one_config_from_X)
from spotPython.plot.xai import (get_activations, get_gradients, get_weights,
                                 plot_nn_values_hist, plot_nn_values_scatter, visualize_weights,
                                 visualize_gradients, visualize_activations, visualize_gradient_distributions,
                                 visualize_weights_distributions)
from spotPython.light.predictmodel import predict_model
from spotPython.utils.file import save_experiment, load_experiment
from spotPython.data.lightdatamodule import LightDataModule
from spotPython.light.regression.netlightregression2 import NetLightRegression2
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from pyhcf.data.loadHcfData import loadFeaturesFromPkl
from pyhcf.data.param_list_generator import load_relevante_aero_variablen
from spotPython.hyperparameters.values import set_control_key_value
from spotPython.data.pkldataset import PKLDataset
import torch



def run_spot_python_experiment(
    MAX_TIME=1,
    INIT_SIZE=5,
    PREFIX="0000-spot",
    FUN_EVALS=10,
    FUN_REPEATS=1,
    n_total=None,
    perc_train=0.6,
    data_set="Phishing",
    target="is_phishing",
    filename="PhishingData.csv",
    directory="./userData",
    n_samples=1_250,
    n_features=9,
    coremodel="NetLightRegression2",
    log_level=50,
    DATA_PKL_NAME = "DATA.pickle",
    NOISE = False,
    OCBA_DELTA = 0,
    REPEATS = 2,
    INIT_SIZE = 20,
    WORKERS = 0,
    DEVICE = getDevice(),
    DEVICES = 1,
    TEST_SIZE = 0.3,
    K_FOLDS = 5,
    target = "N"
    rmNA=True
    rmMF=True
    scale_data=True

) -> spot.Spot:
    """Runs a spot experiment.

    """

    fun_control = fun_control_init(
        _L_in=len(param_list),
        _L_out=1,
        PREFIX=PREFIX,
        TENSORBOARD_CLEAN=True,
        device=DEVICE,
        enable_progress_bar=False,
        fun_evals=FUN_EVALS,
        fun_repeats=FUN_REPEATS,
        log_level=50,
        max_time=MAX_TIME,
        num_workers=WORKERS,
        ocba_delta = OCBA_DELTA,
        show_progress=True,
        test_size=TEST_SIZE,
        tolerance_x=np.sqrt(np.spacing(1)),
        verbosity=1,
        noise=NOISE
        )
    
    # Data set  
    dataset = PKLDataset(directory="./userData/",
                        filename="data_sensitive.pkl",
                        target_column=target,
                        feature_type=torch.float32,
                        target_type=torch.float32,
                        rmNA=True)
    set_control_key_value(control_dict=fun_control,
                            key="data_set",
                            value=dataset,
                            replace=True)
    print(len(dataset))

    add_core_model_to_fun_control(fun_control=fun_control,
                              core_model=NetLightRegression2,
                              hyper_dict=LightHyperDict)
    print(gen_design_table(fun_control))

    fun = HyperLight(log_level=50).fun

    design_control = design_control_init(init_size=INIT_SIZE,
                                        repeats=REPEATS,)

    surrogate_control = surrogate_control_init(noise=True,
                                                n_theta=2,
                                                min_Lambda=1e-6,
                                                max_Lambda=10,
                                                log_level=50,)

    spot_tuner = spot.Spot(fun=fun,
                        fun_control=fun_control,
                        design_control=design_control,
                        surrogate_control=surrogate_control)
    spot_tuner.run()

    SPOT_PKL_NAME = save_experiment(spot_tuner, fun_control)

    # tensorboard --logdir="runs/"

    
    X_start = get_default_hyperparameters_as_array(fun_control)
    fun = HyperRiver(log_level=fun_control["log_level"]).fun_oml_horizon

    design_control = design_control_init()
    set_control_key_value(control_dict=design_control, key="init_size", value=INIT_SIZE, replace=True)

    surrogate_control = surrogate_control_init(noise=True, n_theta=2)

    p_open = start_tensorboard()
    print(fun_control)

    spot_tuner = spot.Spot(
        fun=fun, fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control
    )
    spot_tuner.run(X_start=X_start)

    # stop_tensorboard(p_open)
    return spot_tuner, fun_control


def compare_tuned_default(spot_tuner, fun_control) -> None:
    X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1, -1))
    print(f"X = {X}")
    model_spot = get_one_core_model_from_X(X, fun_control)
    df_eval_spot, df_true_spot = eval_oml_horizon(
        model=model_spot,
        train=fun_control["train"],
        test=fun_control["test"],
        target_column=fun_control["target_column"],
        horizon=fun_control["horizon"],
        oml_grace_period=fun_control["oml_grace_period"],
        metric=fun_control["metric_sklearn"],
    )
    X_start = get_default_hyperparameters_as_array(fun_control)
    model_default = get_one_core_model_from_X(X_start, fun_control)
    df_eval_default, df_true_default = eval_oml_horizon(
        model=model_default,
        train=fun_control["train"],
        test=fun_control["test"],
        target_column=fun_control["target_column"],
        horizon=fun_control["horizon"],
        oml_grace_period=fun_control["oml_grace_period"],
        metric=fun_control["metric_sklearn"],
    )

    df_labels = ["default", "spot"]

    # First Plot

    plot_bml_oml_horizon_metrics(
        df_eval=[df_eval_default, df_eval_spot],
        log_y=False,
        df_labels=df_labels,
        metric=fun_control["metric_sklearn"],
        filename=None,
        show=False,
    )
    plt.figure(1)

    # Second Plot
    plot_roc_from_dataframes(
        [df_true_default, df_true_spot],
        model_names=["default", "spot"],
        target_column=fun_control["target_column"],
        show=False,
    )
    plt.figure(2)
    # Third Plot

    plot_confusion_matrix(
        df=df_true_default,
        title="Default",
        y_true_name=fun_control["target_column"],
        y_pred_name="Prediction",
        show=False,
    )
    plt.figure(2)
    # Fourth Plot

    plot_confusion_matrix(
        df=df_true_spot, title="Spot", y_true_name=fun_control["target_column"], y_pred_name="Prediction", show=False
    )
    plt.figure(3)

    plt.show()  # Display all four plots simultaneously


def parallel_plot(spot_tuner):
    fig = spot_tuner.parallel_plot()
    fig.show()


def contour_plot(spot_tuner):
    spot_tuner.plot_important_hyperparameter_contour(show=False)
    pylab.show()


def importance_plot(spot_tuner):
    plt.figure()
    spot_tuner.plot_importance(show=False)
    plt.show()


def progress_plot(spot_tuner):
    spot_tuner.plot_progress(show=False)
    plt.show()
