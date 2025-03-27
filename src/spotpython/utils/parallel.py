from multiprocessing import Pool, Manager
from joblib import Parallel, delayed
import numpy as np
from typing import Callable, Any, Union


def evaluate_row(row: Union[np.ndarray, list], objective_function: Callable[[np.ndarray, Any], Any], fun_control: Any) -> Any:
    """
    Evaluates a single row using the provided objective function.

    Args:
        row (array-like): The input data for the row to be evaluated.
        objective_function (callable): A function that computes the objective value.
            It should accept a NumPy array and an additional control parameter.
        fun_control (any): Additional control parameter to be passed to the objective function.

    Returns:
        The result of the objective function applied to the row.

    Examples:
        >>> from spotpython.utils.parallel import evaluate_row
        >>> import numpy as np
        >>> def sample_objective(row, control):
        ...     return sum(row) + control.get('offset', 0)
        >>> row = [1, 2, 3]
        >>> fun_control = {'offset': 10}
        >>> evaluate_row(row, sample_objective, fun_control)
            array([11, 12, 13])
    """
    return objective_function(np.array([row]), fun_control)


def parallel_objective_function(objective_function, X, num_cores, fun_control, method) -> np.ndarray:
    """
    Executes an objective function in parallel using either multiprocessing or joblib.

    Args:
        objective_function (callable): The function to be evaluated for each row in `X`.
        X (iterable): The input data, where each element represents a row to be processed.
        num_cores (int): The number of CPU cores to use for parallel processing.
        fun_control (dict): A dictionary of shared control parameters for the objective function.
        method (str): The parallelization method to use. Options are:
            - 'mp': Use Python's multiprocessing module.
            - 'joblib': Use the joblib library.

    Returns:
        numpy.ndarray: A flattened array of results obtained by applying the objective function to each row in `X`.

    Raises:
        ValueError: If an unsupported `method` is provided.

    Examples:
        >>> from spotpython.utils.parallel import parallel_objective_function
        >>> import numpy as np
        >>> def sample_objective(row, control):
        ...     return sum(row) + control.get('offset', 0)
        >>> X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> fun_control = {'offset': 10}
        >>> parallel_objective_function(sample_objective, X, num_cores=2, fun_control=fun_control, method='mp')
        array([16, 25, 34])
        >>> parallel_objective_function(sample_objective, X, num_cores=2, fun_control=fun_control, method='joblib')
        array([16, 25, 34])
    """
    with Manager() as manager:
        shared_control = manager.dict(fun_control)
        if method == "mp":
            with Pool(processes=num_cores) as pool:
                results = pool.starmap(evaluate_row, [(row, objective_function, shared_control) for row in X])
        elif method == "joblib":
            results = Parallel(n_jobs=num_cores)(delayed(evaluate_row)(row, objective_function, shared_control) for row in X)

    return np.array(results).flatten()


def make_parallel(obj_func, num_cores, method="mp") -> Callable:
    """
    Creates a parallelized wrapper function for the given objective function.

    Args:
        obj_func (callable): The objective function to be parallelized.
            It should accept the same arguments as the wrapper function.
        num_cores (int): The number of cores to use for parallel processing.
        method (str, optional): The parallelization method to use.
            Defaults to 'mp' (multiprocessing). Other methods may be supported
            depending on the implementation of `parallel_objective_function`.

    Returns:
        callable: A wrapper function that executes the objective function
        in parallel using the specified number of cores and method.

    Examples:
        >>> from spotpython.utils.parallel import make_parallel
        >>> def sample_function(x):
        ...     return x ** 2
        ...
        >>> parallel_func = make_parallel(sample_function, num_cores=4, method='mp')
        >>> result = parallel_func([1, 2, 3, 4])
        >>> print(result)
        [1, 4, 9, 16]
    """
    global parallel_wrap

    def parallel_wrap(X, fun_control=None):
        return parallel_objective_function(obj_func, X, num_cores, fun_control, method)

    return parallel_wrap
