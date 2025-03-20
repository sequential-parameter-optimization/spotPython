from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import numpy as np
import pandas as pd
from typing import List, Tuple, Union


def get_num_cols(df: pd.DataFrame) -> list:
    """
    Identifies numerical columns in a DataFrame.

    This function selects columns with numerical data types (e.g., int, float)
    from the given DataFrame and returns their names as a list.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names corresponding to numerical columns.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     "age": [25, 30, np.nan, 35],
        ...     "gender": ["M", "F", "M", "F"],
        ...     "income": [50000, 60000, 55000, np.nan]
        ... })
        >>> get_num_cols(df)
        ['age', 'income']
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_cat_cols(df: pd.DataFrame) -> list:
    """
    Identifies categorical columns in a DataFrame.

    This function selects columns with object data types (e.g., strings)
    or columns with all NaN values from the given DataFrame and returns their names as a list.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names corresponding to categorical columns.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     "age": [25, 30, np.nan, 35],
        ...     "gender": ["M", "F", "M", "F"],
        ...     "income": [50000, 60000, 55000, np.nan]
        ... })
        >>> get_cat_cols(df)
        ['gender']
    """
    return df.select_dtypes(include=["object"]).columns.tolist() + [col for col in df.columns if df[col].isna().all()]


def generic_preprocess_df(
    df: pd.DataFrame,
    target: Union[str, List[str]],
    imputer_num=SimpleImputer(strategy="mean"),
    imputer_cat=SimpleImputer(strategy="most_frequent"),
    encoder_cat=OneHotEncoder(categories="auto", drop=None, handle_unknown="ignore", sparse_output=False),
    scaler_num=RobustScaler(),
    test_size=0.2,
    random_state=42,
    shuffle=True,
    n_jobs=None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses a DataFrame by handling numerical and categorical features,
    splitting the data into training and testing sets, and applying transformations.
    Supports single or multiple target columns.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
        target (Union[str, List[str]]): The name(s) of the target column(s) to predict.
            Can be a single string or a list of strings.
        imputer_num (SimpleImputer, optional): Imputer for numerical columns.
            Defaults to `SimpleImputer(strategy="mean")`.
        imputer_cat (SimpleImputer, optional): Imputer for categorical columns.
            Defaults to `SimpleImputer(strategy="most_frequent")`.
        encoder_cat (OneHotEncoder, optional): Encoder for categorical columns.
            Defaults to `OneHotEncoder(categories="auto", drop=None, handle_unknown="ignore")`.
        scaler_num (RobustScaler, optional): Scaler for numerical columns.
            Defaults to `RobustScaler()`.
        test_size (float, optional): Proportion of the dataset to include in the test split.
            Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
        n_jobs (int, optional): Number of jobs to run in parallel for the `ColumnTransformer`.
            Defaults to None (1 job).

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
            A tuple containing:
            - X_train (np.ndarray): Transformed training feature set.
            - X_test (np.ndarray): Transformed testing feature set.
            - y_train (pd.DataFrame): Training target values.
            - y_test (pd.DataFrame): Testing target values.

    Raises:
        ValueError: If any of the target column(s) are not found in the DataFrame.

    Examples:
        >>> from spotpython.utils.preprocess import generic_preprocess_df
        >>> import pandas as pd
        >>> from sklearn.impute import SimpleImputer
        >>> from sklearn.preprocessing import OneHotEncoder, RobustScaler
        >>> df = pd.DataFrame({
        ...     "age": [25, 30, np.nan, 35],
        ...     "gender": ["M", "F", "M", "F"],
        ...     "income": [50000, 60000, 55000, np.nan],
        ...     "target1": [1, 0, 1, 0],
        ...     "target2": [0, 1, 0, 1]
        ... })
        >>> X_train, X_test, y_train, y_test = generic_preprocess_df(
        ...     df,
        ...     target=["target1", "target2"],
        ...     imputer_num=SimpleImputer(strategy="mean"),
        ...     imputer_cat=SimpleImputer(strategy="most_frequent"),
        ...     encoder_cat=OneHotEncoder(),
        ...     scaler_num=RobustScaler(),
        ...     test_size=0.25,
        ...     random_state=42
        ... )
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    if isinstance(target, str):
        target = [target]  # Convert to list for consistent handling

    for t in target:
        if t not in df.columns:
            raise ValueError(f"Target column '{t}' not found in the DataFrame.")

    X = df.drop(target, axis=1)
    y = df[target]

    num_cols = get_num_cols(X)
    cat_cols = get_cat_cols(X)
    X[cat_cols] = X[cat_cols].astype(str)

    numerical_transformer = Pipeline(steps=[("imputer", imputer_num), ("scaler", scaler_num)])
    categorical_transformer = Pipeline(steps=[("imputer", imputer_cat), ("encoder", encoder_cat)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, num_cols),
            ("categorical", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
        n_jobs=n_jobs,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test
