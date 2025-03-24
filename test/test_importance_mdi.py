import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from spotpython.plot.importance import generate_mdi

def test_generate_mdi_with_dataframe():
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    # Call the function
    result = generate_mdi(X_df, y_series)

    # Assertions
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert list(result.columns) == ["Feature", "Importance"], "DataFrame should have 'Feature' and 'Importance' columns"
    assert len(result) == X_df.shape[1], "Number of rows should match the number of features"
    assert result["Importance"].sum() > 0, "Feature importances should be greater than zero"

def test_generate_mdi_with_ndarray():
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    # Call the function
    result = generate_mdi(X, y)

    # Assertions
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert list(result.columns) == ["Feature", "Importance"], "DataFrame should have 'Feature' and 'Importance' columns"
    assert len(result) == X.shape[1], "Number of rows should match the number of features"
    assert result["Importance"].sum() > 0, "Feature importances should be greater than zero"

def test_generate_mdi_with_custom_feature_names():
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    feature_names = [f"Custom_Feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X)

    # Call the function
    result = generate_mdi(X_df, y, feature_names=feature_names)

    # Assertions
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert list(result.columns) == ["Feature", "Importance"], "DataFrame should have 'Feature' and 'Importance' columns"
    assert len(result) == len(feature_names), "Number of rows should match the number of custom feature names"
    assert set(result["Feature"].values) == set(feature_names), "Feature names should match the custom feature names"
    assert result["Importance"].sum() > 0, "Feature importances should be greater than zero"