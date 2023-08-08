import numpy as np
from spotPython.utils.convert import get_Xy_from_df
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer
from spotPython.utils.metrics import mapk_scorer


def evaluate_model(model, fun_control):
    try:
        X_train, y_train = get_Xy_from_df(fun_control["train"], fun_control["target_column"])
        X_test, y_test = get_Xy_from_df(fun_control["test"], fun_control["target_column"])
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


def evaluate_hold_out(model, fun_control):
    train_df = fun_control["train"]
    target_column = fun_control["target_column"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            train_df.drop(target_column, axis=1),
            train_df[target_column],
            random_state=42,
            test_size=0.25,
            stratify=train_df[target_column],
        )
        model.fit(X_train, y_train)
        # convert to numpy array, see https://github.com/scikit-learn/scikit-learn/pull/26772
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        if fun_control["predict_proba"]:
            df_preds = model.predict_proba(X_test)
        else:
            df_preds = model.predict(X_test)
        df_eval = fun_control["metric_sklearn"](y_test, df_preds, **fun_control["metric_params"])
    except Exception as err:
        print(f"Error in fun_sklearn(). Call to evaluate_hold_out failed. {err=}, {type(err)=}")
        df_eval = np.nan
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
