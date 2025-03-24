import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from spotpython.plot.importance import plot_importances

@pytest.fixture
def sample_data():
    # Generate sample data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f"Feature_{i}" for i in range(5)])
    X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"Feature_{i}" for i in range(5)])
    y_train = pd.Series(np.random.rand(100))
    y_test = pd.Series(np.random.rand(20))
    return X_train, X_test, y_train, y_test

@pytest.fixture
def mdi_importances(sample_data):
    # Generate MDI importances
    X_train, _, y_train, _ = sample_data
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    df_mdi = pd.DataFrame({"Feature": X_train.columns, "Importance": importances}).sort_values("Importance", ascending=False)
    return df_mdi

@pytest.fixture
def perm_importances(sample_data):
    # Generate permutation importances
    X_train, X_test, y_train, y_test = sample_data
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    return perm_imp

def test_plot_importances(sample_data, mdi_importances, perm_importances):
    X_train, X_test, y_train, y_test = sample_data
    df_mdi = mdi_importances
    perm_imp = perm_importances

    # Test if the function runs without errors
    try:
        plot_importances(df_mdi, perm_imp, X_test, target_name="Test Target", feature_names=X_train.columns, k=3, show=False)
    except Exception as e:
        pytest.fail(f"plot_importances raised an exception: {e}")