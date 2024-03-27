import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that knows how to encode numpy arrays.

    Note:
        Taken from:
        https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            if np.isnan(obj):
                return "NaN"
            elif np.isinf(obj):
                return "Inf"
            else:
                return float(obj)
        return json.JSONEncoder.default(self, obj)
