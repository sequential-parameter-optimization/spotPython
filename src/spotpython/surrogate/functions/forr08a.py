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
        >>> from spotpython.surrogate.functions.branin import branin
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
