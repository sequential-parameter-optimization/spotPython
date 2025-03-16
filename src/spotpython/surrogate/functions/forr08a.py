import numpy as np


def onevar(x: np.ndarray) -> np.ndarray:
    """
    One-variable test function that takes a scalar or 1D array input `x` in the range [0, 1]
    and returns the corresponding function values. The function is vectorized to handle
    multiple inputs.

    The function is defined as:
        f(x) = ((6x - 2)^2) * np.sin((6x - 2) * 2)

    Args:
        x (np.ndarray): A scalar or 1D NumPy array of values in the range [0, 1].

    Returns:
        np.ndarray: The calculated function values for the input `x`.

    Raises:
        ValueError: If any value in `x` is outside the range [0, 1].

    Examples:
        >>> import numpy as np
        >>> from spotpython.surrogate.functions.forr08a import onevar
        >>> # Single input
        >>> print(onevar(np.array([0.5])))
        [0.9093]
        >>> # Multiple inputs
        >>> x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        >>> print(onevar(x))
        [3.0272, -0.2104, 0.9093,  -5.9933, 15.8297]
    """
    # Ensure x is a NumPy array
    x = np.asarray(x)

    # Check if all values are within the range [0, 1]
    if np.any((x < 0) | (x > 1)):
        raise ValueError("Variable outside of range - use x in [0, 1]")

    # Compute the function values
    f = ((6 * x - 2) ** 2) * np.sin((6 * x - 2) * 2)

    return f


def branin(x: np.ndarray) -> np.ndarray:
    """
    Branin's test function that takes a 2D input vector `x` in the range [0, 1] for each dimension
    and returns the corresponding scalar function value. The function is vectorized to handle
    multiple inputs.

    The function is defined as:
        f(x) = a * (X2 - b * X1^2 + c * X1 - d)^2 + e * (1 - ff) * cos(X1) + e + 5 * x1
    where:
        X1 = 15 * x1 - 5
        X2 = 15 * x2

    Args:
        x (np.ndarray): A 2D NumPy array of shape (n_samples, 2) where each row is a 2D input vector.

    Returns:
        np.ndarray: The calculated function values for the input `x`.

    Raises:
        ValueError: If `x` does not have exactly 2 columns or if any value in `x` is outside the range [0, 1].

    Examples:
        >>> import numpy as np
        >>> from spotpython.surrogate.functions.forr08a import branin
        >>> # Single input
        >>> print(branin(np.array([[0.5, 0.5]])))
        [26.63]
        >>> # Multiple inputs
        >>> x = np.array([[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]])
        >>> print(branin(x))
        [308.1291, 34.0028, 26.63, 126.3879, 150.8722]
    """
    # Ensure x is a NumPy array
    x = np.asarray(x)

    # Check if x has exactly 2 columns
    if x.shape[1] != 2:
        raise IndexError("Input to branin must have exactly 2 columns.")

    # Check if all values are within the range [0, 1]
    if np.any((x < 0) | (x > 1)):
        raise ValueError("Variable outside of range - use x in [0, 1] for both dimensions.")

    # Extract x1 and x2
    x1, x2 = x[:, 0], x[:, 1]

    # Transform x1 and x2 to X1 and X2
    X1 = 15 * x1 - 5
    X2 = 15 * x2

    # Define constants
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    d = 6
    e = 10
    ff = 1 / (8 * np.pi)

    # Compute the function values
    f = a * (X2 - b * X1**2 + c * X1 - d) ** 2 + e * (1 - ff) * np.cos(X1) + e + 5 * x1

    return f


def aerofoilcd(X: np.ndarray) -> np.ndarray:
    """
    Computes the drag coefficient (cd) of an aerofoil based on the shape parameter X.

    This function reads the drag coefficient data from the "cd_data.csv" file and uses
    the input X (rounded to the nearest 0.01) to return the corresponding drag coefficients.

    Args:
        X (np.ndarray): A 1D NumPy array of values in the range [0, 1] representing the shape parameters.

    Returns:
        np.ndarray: A 1D NumPy array of drag coefficients (cd) corresponding to the input X.

    Raises:
        ValueError: If any value in X is outside the range [0, 1].

    Examples:
        >>> from spotpython.surrogate.functions.forr08a import aerofoilcd
        >>> X = np.array([0.5, 0.75])
        >>> aerofoilcd(X)
        array([0.029975, 0.033375])
    """
    # Ensure X is a NumPy array
    X = np.asarray(X)

    # Validate the input
    if np.any((X < 0) | (X > 1)):
        raise ValueError("All values in X must be in the range [0, 1].")

    # The given data as a string
    data = (
        "0.031745,0.031568,0.031355,0.031607,0.03132,0.031242,0.030959,0.030593,0.030347,"
        "0.030153,0.030089,0.029881,0.029967,0.029686,0.029612,0.029727,0.029445,0.030188,"
        "0.029907,0.029634,0.02978,0.029585,0.029301,0.029543,0.029663,0.029137,0.029611,"
        "0.029395,0.02918,0.029369,0.029272,0.029384,0.029249,0.029545,0.029641,0.029975,"
        "0.029801,0.029857,0.030131,0.029678,0.029451,0.029899,0.029922,0.030228,0.02979,"
        "0.03004,0.030188,0.030366,0.030399,0.030193,0.030012,0.030109,0.030629,0.030551,"
        "0.030721,0.031211,0.031132,0.031236,0.031379,0.031531,0.03117,0.031808,0.0318,"
        "0.032141,0.032216,0.032451,0.032545,0.032836,0.032843,0.032888,0.033098,0.033271,"
        "0.033478,0.03328,0.033375,0.033979,0.034197,0.034406,0.034315,0.034662,0.035125,"
        "0.035306,0.035021,0.03526,0.035988,0.03579,0.036927,0.036705,0.037232,0.037563,"
        "0.037501,0.037802,0.038302,0.038676,0.038898,0.03891,0.03916,0.039584,0.038509,"
        "0.040168,0.039062"
    )

    # Convert the string to a NumPy array
    cd_data = np.fromstring(data, sep=",")

    # Compute the indices based on X (rounded to the nearest 0.01)
    indices = np.round(X * 100).astype(int)

    # Return the corresponding drag coefficients
    return cd_data[indices]
