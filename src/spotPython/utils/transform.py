def scale(X, lower, upper):
    """
    Sample scaling from unit hypercube to different bounds.

    Converts a sample from `[0, 1)` to `[a, b)`.
    Note: equal lower and upper bounds are feasible.
    The following transformation is used:

    `(b - a) * X + a`

    Args:
    X (array):
        Sample to scale.
    lower (array):
        lower bound of transformed data.
    upper (array):
        upper bounds of transformed data.

    Returns:
    (array):
        Scaled sample.

    Examples:
    Transform three samples in the unit hypercube to (lower, upper) bounds:

    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> from spotPython.utils.transform import scale
    >>> lower = np.array([6, 0])
    >>> upper = np.array([6, 5])
    >>> sample = np.array([[0.5 , 0.75],
    >>>             [0.5 , 0.5],
    >>>             [0.75, 0.25]])
    >>> scale(sample, lower, upper)

    """
    # Checking that X is within (0,1) interval
    if (X.max() > 1.0) or (X.min() < 0.0):
        raise ValueError("Sample is not in unit hypercube")

    for i in range(X.shape[1]):
        if lower[i] == upper[i]:
            X[:, i] = lower[i]
        else:
            X[:, i] = X[:, i] * (upper[i] - lower[i]) + lower[i]
    return X
