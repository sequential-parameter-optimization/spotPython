import numpy as np
from sklearn.gaussian_process.kernels import Kernel, NormalizedKernelMixin, Hyperparameter


def _correlation(kernel, D, kernel_params=None):
    """
    Dispatches to the selected kernel function.
    Args:
        kernel: Kernel type (str, callable, or sklearn-compatible kernel object)
        D (np.ndarray): Distance matrix.
        kernel_params (dict): Parameters for the kernel (optional).
    Returns:
        np.ndarray: Correlation matrix.
    """
    kernel_params = kernel_params or {}
    # If kernel is a sklearn-compatible kernel object
    if isinstance(kernel, Kernel):
        # These expect X, Y, not D, so we call on D as before for compatibility
        return kernel(D)
    elif callable(kernel):
        return kernel(D, **kernel_params)
    elif kernel == "gauss":
        return np.exp(-D)
    elif kernel == "matern":
        nu = kernel_params.get("nu", 2.5)
        if nu == 0.5:
            return np.exp(-np.sqrt(D))
        elif nu == 1.5:
            sqrt3D = np.sqrt(3.0 * D)
            return (1.0 + sqrt3D) * np.exp(-sqrt3D)
        elif nu == 2.5:
            sqrt5D = np.sqrt(5.0 * D)
            return (1.0 + sqrt5D + (5.0 / 3.0) * D) * np.exp(-sqrt5D)
        else:
            return np.exp(-D)
    elif kernel == "exp":
        return np.exp(-np.sqrt(D))
    elif kernel == "cubic":
        return 1.0 - D**3
    elif kernel == "linear":
        return 1.0 - D
    elif kernel == "rq":
        alpha = kernel_params.get("alpha", 1.0)
        return (1.0 + D / (2.0 * alpha)) ** (-alpha)
    elif kernel == "poly":
        degree = kernel_params.get("degree", 2)
        return (1.0 + D) ** degree
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


# Example: Custom sklearn-compatible RBF kernel using _correlation
class CustomRBF(NormalizedKernelMixin, Kernel):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, D, eval_gradient=False):
        # D is assumed to be the squared distance matrix
        K = _correlation("gauss", D)
        if eval_gradient:
            # Gradient not implemented
            return K, np.empty((K.shape[0], K.shape[1], 0))
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True


class CustomMatern(NormalizedKernelMixin, Kernel):
    def __init__(self, length_scale=1.0, nu=1.5, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.nu = nu
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, D, eval_gradient=False):
        K = _correlation("matern", D, {"nu": self.nu})
        if eval_gradient:
            # Gradient not implemented
            return K, np.empty((K.shape[0], K.shape[1], 0))
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True
