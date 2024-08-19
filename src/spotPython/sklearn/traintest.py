import numpy as np
from spotpython.utils.convert import get_Xy_from_df
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer
from spotpython.utils.metrics import mapk_scorer
import pandas as pd


def evaluate_model(model, fun_control) -> np.ndarray:
    """Evaluate a model using the test set.
    First, the model is trained on the training set. If a scaler
    is provided, the data is transformed using the scaler and `fit_transform(X_train)`.
    Then, the model is evaluated using the test set from `fun_control`,
    the scaler with `transform(X_test)`,
    the model.predict() method and the
    `metric_params` specified in `fun_control`.

    Note:
    In contrast to `evaluate_hold_out()`, this function uses the test set.
    It can be selected by setting `fun_control["eval"] = "eval_test"`.

    Args:
        model (sklearn model):
            sklearn model.
        fun_control (dict):
            dictionary containing control parameters for the function.

    Returns:
        (np.ndarray): array containing evaluation results.
    """
    try:
        X_train, y_train = get_Xy_from_df(fun_control["train"], fun_control["target_column"])
        X_test, y_test = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
        if fun_control["scaler"] is not None:
            X_train = fun_control["scaler"]().fit_transform(X_train)
            X_test = fun_control["scaler"]().transform(X_test)
        model.fit(X_train, y_train)
        if fun_control["predict_proba"]:
            df_preds = model.predict_proba(X_test)
        else:
            df_preds = model.predict(X_test)
        df_eval = fun_control["metric_sklearn"](y_test, df_preds, **fun_control["metric_params"])
    except Exception as err:
        print(f"Error in fun_sklearn(). Call to evaluate_model failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_eval = np.nan
    return df_eval, df_preds


def evaluate_hold_out(model, fun_control) -> np.ndarray:
    """Evaluate a model using hold-out validation.
    A validation set is created from the training set.
    The test set is not used in this evaluation.

    Note:
    In contrast to `evaluate_model()`, this function creates a validation set as
    a subset of the training set.
    It can be selected by setting `fun_control["eval"] = "evaluate_hold_out"`.

    Args:
        model (sklearn model):
            sklearn model.
        fun_control (dict):
            dictionary containing control parameters for the function.

    Returns:
        (np.ndarray): array containing evaluation results.

    Raises:
        Exception: if call to train_test_split() or fit() or predict() fails.
    """
    train_df = fun_control["train"]
    target_column = fun_control["target_column"]
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            train_df.drop(target_column, axis=1),
            train_df[target_column],
            random_state=42,
            test_size=fun_control["test_size"],
            # stratify=train_df[target_column],
        )
    except Exception as err:
        print(f"Error in evaluate_hold_out(). Call to train_test_split() failed. {err=}, {type(err)=}")
    try:
        if fun_control["scaler"] is not None:
            scaler = fun_control["scaler"]()
            X_train = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(
                X_train, columns=train_df.drop(target_column, axis=1).columns
            )  # Maintain column names
        model.fit(X_train, y_train)
    except Exception as err:
        print(f"Error in evaluate_hold_out(). Call to fit() failed. {err=}, {type(err)=}")
    try:
        if fun_control["scaler"] is not None:
            X_val = scaler.transform(X_val)
            X_val = pd.DataFrame(X_val, columns=train_df.drop(target_column, axis=1).columns)  # Maintain column names
        y_val = np.array(y_val)
        if fun_control["predict_proba"] or fun_control["task"] == "classification":
            df_preds = model.predict_proba(X_val)
        else:
            df_preds = model.predict(X_val)
        df_eval = fun_control["metric_sklearn"](y_val, df_preds, **fun_control["metric_params"])
    except Exception as err:
        print(f"Error in evaluate_hold_out(). Call to predict() failed. {err=}, {type(err)=}")
        df_eval = np.nan
    return df_eval, df_preds


def evaluate_cv(model, fun_control, verbose=0):
    if fun_control["eval"] == "train_cv":
        X, y = get_Xy_from_df(fun_control["train"], fun_control["target_column"])
    elif fun_control["eval"] == "test_cv":
        X, y = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
    else:  # full dataset
        X, y = get_Xy_from_df(fun_control["data"], fun_control["target_column"])
    k = fun_control["k_folds"]
    try:
        if fun_control["predict_proba"]:
            # TODO: add more scorers
            # proba_scorer = make_scorer(fun_control["metric_sklearn"], **fun_control["metric_params"])
            proba_scorer = mapk_scorer
            scores = cross_val_score(model, X, y, cv=k, scoring=proba_scorer, verbose=verbose)
        else:
            sklearn_scorer = make_scorer(fun_control["metric_sklearn"], **fun_control["metric_params"])
            scores = cross_val_score(model, X, y, cv=k, scoring=sklearn_scorer)
        df_eval = scores.mean()
    except Exception as err:
        print(f"Error in fun_sklearn(). Call to evaluate_cv failed. {err=}, {type(err)=}")
        df_eval = np.nan
    return df_eval, None


def evaluate_model_oob(model, fun_control):
    """Out-of-bag evaluation (Only for RandomForestClassifier).
    If fun_control["eval"] == "eval_oob_score".
    """
    try:
        X, y = get_Xy_from_df(fun_control["train"], fun_control["target_column"])
        model.fit(X, y)
        df_preds = model.oob_decision_function_
        df_eval = fun_control["metric_sklearn"](y, df_preds, **fun_control["metric_params"])
    except Exception as err:
        print(f"Error in fun_sklearn(). Call to evaluate_model_oob failed. {err=}, {type(err)=}")
        df_eval = np.nan
        df_eval = np.nan
    return df_eval, df_preds
