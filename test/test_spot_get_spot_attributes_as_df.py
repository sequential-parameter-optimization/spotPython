import numpy as np
import pytest
import pandas as pd

from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def dummy_fun(X, fun_control=None):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)

def test_get_spot_attributes_as_df_basic():
    fun_control = fun_control_init(lower=np.array([-1, -1]), upper=np.array([1, 1]), fun_evals=5)
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=dummy_fun, fun_control=fun_control, design_control=design_control)
    df = spot.get_spot_attributes_as_df()
    assert isinstance(df, pd.DataFrame)
    # Check that some expected attributes are present
    assert "fun_control" in df["Attribute Name"].values
    assert "design_control" in df["Attribute Name"].values
    assert "surrogate" in df["Attribute Name"].values
    # Check that the number of rows matches the number of attributes
    assert len(df) == len([attr for attr in dir(spot) if not callable(getattr(spot, attr)) and not attr.startswith("__")])

def test_get_spot_attributes_as_df_contains_values():
    fun_control = fun_control_init(lower=np.array([-2, -2]), upper=np.array([2, 2]), fun_evals=3)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    df = spot.get_spot_attributes_as_df()
    # Check that lower and upper bounds are present and correct
    lower_row = df[df["Attribute Name"] == "lower"]
    upper_row = df[df["Attribute Name"] == "upper"]
    assert np.allclose(lower_row.iloc[0]["Attribute Value"], np.array([-2, -2]))
    assert np.allclose(upper_row.iloc[0]["Attribute Value"], np.array([2, 2]))

def test_get_spot_attributes_as_df_dataframe_content():
    fun_control = fun_control_init(lower=np.array([0]), upper=np.array([1]), fun_evals=2)
    spot = Spot(fun=dummy_fun, fun_control=fun_control)
    df = spot.get_spot_attributes_as_df()
    # DataFrame should have columns 'Attribute Name' and 'Attribute Value'
    assert set(df.columns) == {"Attribute Name", "Attribute Value"}
    # There should be at least one attribute
    assert len(df) > 0