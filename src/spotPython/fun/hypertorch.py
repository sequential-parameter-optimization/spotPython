from numpy.random import default_rng
import numpy as np
from numpy import array
from sklearn.pipeline import make_pipeline
from spotPython.utils.convert import get_Xy_from_df
from spotPython.utils.data import load_data
import torch.nn as nn
import torch.optim as optim
import torch
import os
from torch.utils.data import random_split


from spotPython.hyperparameters.values import assign_values
from spotPython.hyperparameters.prepare import (
    get_one_config_from_var_dict,
)

import logging
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)
# configure the handler and formatter as needed
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
# add formatter to the handler
py_handler.setFormatter(py_formatter)
# add handler to the logger
logger.addHandler(py_handler)


class HyperTorch:
    """
    Hyperparameter Tuning for Torch.

    Args:
        seed (int): seed.
            See [Numpy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start)

    """

    def __init__(self, seed=126, log_level=50):
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.fun_control = {
            "seed": None,
            "data": None,
            "step": 10_000,
            "horizon": None,
            "grace_period": None,
            "metric": None,
            "metric_sklearn": mean_absolute_error,
            "weights": array([1, 0, 0]),
            "weight_coeff": 0.0,
            "log_level": log_level,
            "var_name": [],
            "var_type": [],
            "prep_model": None,
        }
        self.log_level = self.fun_control["log_level"]
        logger.setLevel(self.log_level)
        logger.info(f"Starting the logger at level {self.log_level} for module {__name__}:")

    def check_X_shape(self, X):
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != len(self.fun_control["var_name"]):
            raise Exception

    def evaluate_model(self, model, fun_control):
        # TODO: config anpassen
        try:
            lr = fun_control["lr"]
            checkpoint_dir = fun_control["checkpoint_dir"]
            data_dir = fun_control["data_dir"]

            # X_train, y_train = get_Xy_from_df(fun_control["train"], fun_control["target_column"])
            # X_test, y_test = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
            # model.fit(X_train, y_train)
            # df_preds = model.predict(X_test)
            # df_eval = fun_control["metric_sklearn"](y_test, df_preds)
            #
            device = "cpu"
            # if torch.cuda.is_available():
            #     device = "cuda:0"
            #     if torch.cuda.device_count() > 1:
            #         net = nn.DataParallel(net)
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            # TODO:
            # if checkpoint_dir:
            #     model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            #     model.load_state_dict(model_state)
            #     optimizer.load_state_dict(optimizer_state)

            # TODO:
            # trainset, testset = load_data(data_dir)
            
            trainset = fun_control["train"]

            test_abs = int(len(trainset) * 0.8)
            train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

            trainloader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=int(fun_control["core_model_hyper_dict"]["batch_size"]),
                shuffle=True,
                num_workers=8,
            )
            valloader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=int(fun_control["core_model_hyper_dict"]["batch_size"]),
                shuffle=True,
                num_workers=8,
            )

            for epoch in range(10):  # loop over the dataset multiple times
                running_loss = 0.0
                epoch_steps = 0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    epoch_steps += 1
                    if i % 2000 == 1999:  # print every 2000 mini-batches
                        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                        running_loss = 0.0

                # Validation loss
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0
                pred_list = []
                for i, data in enumerate(valloader, 0):
                    with torch.no_grad():
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        pred_list.append(predicted)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        loss = criterion(outputs, labels)
                        val_loss += loss.cpu().numpy()
                        val_steps += 1

                # TODO:
                # with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            df_eval = val_loss / val_steps
            df_preds = pred_list
            # accuracy = correct / total
        except Exception as err:
            print(f"Error in fun_torch(). Call to evaluate_model failed. {err=}, {type(err)=}")
            df_eval = np.nan
            df_preds = np.nan
        return df_eval, df_preds

    def get_torch_df_eval_preds(self, model):
        try:
            df_eval, df_preds = self.evaluate_model(model, self.fun_control)
        except Exception as err:
            print(f"Error in get_torch_df_eval_preds(). Call to evaluate_model failed. {err=}, {type(err)=}")
            print("Setting df_eval and df.preds to np.nan")
            df_eval = np.nan
            df_preds = np.nan
        return df_eval, df_preds

    def fun_torch(self, X, fun_control=None):
        z_res = np.array([], dtype=float)
        self.fun_control.update(fun_control)
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        for config in get_one_config_from_var_dict(var_dict, self.fun_control):
            if self.fun_control["prep_model"] is not None:
                model = make_pipeline(self.fun_control["prep_model"], self.fun_control["core_model"](**config))
            else:
                model = self.fun_control["core_model"](**config)
            try:
                df_eval, _ = self.evaluate_model(model, self.fun_control)
            except Exception as err:
                print(f"Error in fun_torch(). Call to evaluate_model failed. {err=}, {type(err)=}")
                print("Setting df_eval to np.nan")
                df_eval = np.nan
            z_res = np.append(z_res, fun_control["weights"] * df_eval)
        return z_res
