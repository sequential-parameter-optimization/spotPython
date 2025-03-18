import pandas as pd
import numpy as np
from spotpython.utils.preprocess import get_num_cols
from spotpython.utils.preprocess import get_cat_cols
from spotpython.utils.preprocess import generic_preprocess_df
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def test_generic_preprocess_df_with_only_categorical_data():
    df = pd.DataFrame({
        "country": ["DE", "EN", "DE", "DE", "EN", "DE"],
        "gender": ["M", "F", "M", "M", "F", "M"],
        "target": [1, 0, 1, 1, 0, 1]
    })
    X_train, X_test, y_train, y_test = generic_preprocess_df(
        df,
        target="target",
        imputer_cat=SimpleImputer(strategy="most_frequent"),
        encoder_cat=OneHotEncoder(),
        test_size=0.5,
        random_state=42,
        shuffle=False
    )
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 3
    assert y_train.shape[0] == 3
    assert y_test.shape[0] == 3


def test_get_num_cols_with_mixed_data():
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 35],
        "gender": ["M", "F", "M", "F"],
        "income": [50000, 60000, 55000, np.nan]
    })
    expected = ["age", "income"]
    assert get_num_cols(df) == expected

def test_get_num_cols_with_only_numerical_data():
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "income": [50000, 60000, 55000]
    })
    expected = ["age", "income"]
    assert get_num_cols(df) == expected

def test_get_num_cols_with_only_categorical_data():
    df = pd.DataFrame({
        "gender": ["M", "F", "M"],
        "city": ["NY", "LA", "SF"]
    })
    expected = []
    assert get_num_cols(df) == expected

def test_get_num_cols_with_empty_dataframe():
    df = pd.DataFrame()
    expected = []
    assert get_num_cols(df) == expected

def test_get_num_cols_with_no_numerical_columns():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "city": ["NY", "LA", "SF"]
    })
    expected = []
    assert get_num_cols(df) == expected

def test_get_num_cols_with_all_nan_numerical_columns():
    df = pd.DataFrame({
        "age": [np.nan, np.nan, np.nan],
        "income": [np.nan, np.nan, np.nan]
    })
    expected = ["age", "income"]
    assert get_num_cols(df) == expected

def test_get_cat_cols_with_mixed_data():
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 35],
        "gender": ["M", "F", "M", "F"],
        "income": [50000, 60000, 55000, np.nan]
    })
    expected = ["gender"]
    assert get_cat_cols(df) == expected

def test_get_cat_cols_with_only_categorical_data():
    df = pd.DataFrame({
        "gender": ["M", "F", "M"],
        "city": ["NY", "LA", "SF"]
    })
    expected = ["gender", "city"]
    assert get_cat_cols(df) == expected

def test_get_cat_cols_with_only_numerical_data():
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "income": [50000, 60000, 55000]
    })
    expected = []
    assert get_cat_cols(df) == expected

def test_get_cat_cols_with_empty_dataframe():
    df = pd.DataFrame()
    expected = []
    assert get_cat_cols(df) == expected

def test_get_cat_cols_with_no_categorical_columns():
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "income": [50000, 60000, 55000]
    })
    expected = []
    assert get_cat_cols(df) == expected

def test_generic_preprocess_df_with_mixed_data():
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 35],
        "gender": ["M", "F", "M", "F"],
        "income": [50000, 60000, 55000, np.nan],
        "target": [1, 0, 1, 0]
    })
    X_train, X_test, y_train, y_test = generic_preprocess_df(
        df,
        target="target",
        imputer_num=SimpleImputer(strategy="mean"),
        imputer_cat=SimpleImputer(strategy="most_frequent"),
        encoder_cat=OneHotEncoder(),
        scaler_num=RobustScaler(),
        test_size=0.25,
        random_state=42
    )
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 3
    assert y_test.shape[0] == 1

def test_generic_preprocess_df_with_only_numerical_data():
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "income": [50000, 60000, 55000],
        "target": [1, 0, 1]
    })
    X_train, X_test, y_train, y_test = generic_preprocess_df(
        df,
        target="target",
        imputer_num=SimpleImputer(strategy="mean"),
        scaler_num=RobustScaler(),
        test_size=0.33,
        random_state=42
    )
    assert X_train.shape[0] == 2
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 2
    assert y_test.shape[0] == 1

def test_get_cat_cols_with_all_nan_categorical_columns():
    df = pd.DataFrame({
        "gender": [np.nan, np.nan, np.nan],
        "city": [np.nan, np.nan, np.nan]
    })
    expected = ["gender", "city"]
    assert get_cat_cols(df) == expected


def test_generic_preprocess_df_with_missing_target_column():
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "income": [50000, 60000, 55000]
    })
    try:
        generic_preprocess_df(df, target="target")
    except ValueError as e:
        assert str(e) == "Target column 'target' not found in the DataFrame."


def test_generic_preprocess_df_with_empty_dataframe():
    df = pd.DataFrame()
    try:
        generic_preprocess_df(df, target="target")
    except ValueError as e:
        assert str(e) == "The input DataFrame is empty."