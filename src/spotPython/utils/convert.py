import importlib


def class_for_name(module_name, class_name) -> object:
    """Returns a class for a given module and class name.

    Parameters:
        module_name (str): The name of the module.
        class_name (str): The name of the class.

    Returns:
        object: The class.

    Example:
        >>> from spotPython.utils.convert import class_for_name
            from scipy.optimize import rosen
            bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
            shgo_class = class_for_name("scipy.optimize", "shgo")
            result = shgo_class(rosen, bounds)
    """
    m = importlib.import_module(module_name)
    c = getattr(m, class_name)
    return c
