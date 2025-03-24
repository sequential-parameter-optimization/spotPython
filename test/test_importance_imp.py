import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.utils import Bunch
from spotpython.plot.importance import generate_imp

def test_generate_imp():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Convert to DataFrame/Series for testing
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)

    # Test permutation importance on the test set (default behavior)
    perm_imp_test = generate_imp(X_train_df, X_test_df, y_train_series, y_test_series, use_test=True)
    assert isinstance(perm_imp_test, Bunch), "Output should be a Bunch object"
    assert perm_imp_test.importances_mean.shape[0] == X.shape[1], "Number of importances should match the number of features"
    assert np.all(perm_imp_test.importances_mean >= 0), "All importances should be non-negative"

    # Test permutation importance on the training set
    perm_imp_train = generate_imp(X_train_df, X_test_df, y_train_series, y_test_series, use_test=False)
    assert isinstance(perm_imp_train, Bunch), "Output should be a Bunch object"
    assert perm_imp_train.importances_mean.shape[0] == X.shape[1], "Number of importances should match the number of features"
    assert np.all(perm_imp_train.importances_mean >= 0), "All importances should be non-negative"