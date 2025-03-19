from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from spotpython.surrogate.kriging import Kriging


def get_regressor(name) -> object:
    """
    Returns a scikit-learn regressor based on the given name.

    Args:
        name (str): The name of the regressor.
            Supported names are: "linear", "polynomial", "random_forest", and "kriging".

    Returns:
        object: A scikit-learn regressor object.

    Raises:
        ValueError: If an unknown regressor name is provided.

    Example:
        >>> from spotpython.utils.misc import get_regressor
        >>> regressor = get_regressor("linear")
        >>> print(type(regressor))
        <class 'sklearn.linear_model._base.LinearRegression'>
    """
    if name == "linear":
        mdl = LinearRegression()
    elif name == "polynomial":
        degree_polyn = 2
        mdl = Pipeline([("poly", PolynomialFeatures(degree=degree_polyn)), ("linear", LinearRegression())])
    elif name == "random_forest":
        mdl = RandomForestRegressor()
    # elif name == "xgboost":
    #     mdl = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    elif name == "kriging":
        mdl = Kriging()
    else:
        raise ValueError(f"Unknown regressor {name}")
    return mdl
