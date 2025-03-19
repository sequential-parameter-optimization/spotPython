import numpy as np
import matplotlib.pyplot as plt


class DMax:
    def __init__(self, low, high, scale=1, tol=None):
        if low >= high:
            raise ValueError("The low value must be less than the high value.")
        if scale <= 0:
            raise ValueError("The scale parameter must be greater than zero.")

        self.low = low
        self.high = high
        self.scale = scale
        self.tol = tol
        self.missing = None
        self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        if missing is None:
            missing = self.missing

        out = np.full(len(newdata), np.nan)
        out[newdata < self.low] = 0
        out[newdata > self.high] = 1
        mask = (newdata <= self.high) & (newdata >= self.low)
        out[mask] = ((newdata[mask] - self.low) / (self.high - self.low)) ** self.scale
        out[np.isnan(out)] = missing
        if self.tol is not None:
            out[out == 0] = self.tol
        return out


class DMin:
    def __init__(self, low, high, scale=1, tol=None):
        if low >= high:
            raise ValueError("The low value must be less than the high value.")
        if scale <= 0:
            raise ValueError("The scale parameter must be greater than zero.")

        self.low = low
        self.high = high
        self.scale = scale
        self.tol = tol
        self.missing = None
        self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        if missing is None:
            missing = self.missing

        out = np.full(len(newdata), np.nan)
        out[newdata < self.low] = 1
        out[newdata > self.high] = 0
        mask = (newdata <= self.high) & (newdata >= self.low)
        out[mask] = ((newdata[mask] - self.high) / (self.low - self.high)) ** self.scale
        out[np.isnan(out)] = missing
        if self.tol is not None:
            out[out == 0] = self.tol
        return out


class DTarget:
    def __init__(self, low, target, high, low_scale=1, high_scale=1, tol=None):
        if low >= high:
            raise ValueError("The low value must be less than the high value.")
        if low >= target:
            raise ValueError("The low value must be less than the target.")
        if target >= high:
            raise ValueError("The target value must be less than the high value.")
        if low_scale <= 0 or high_scale <= 0:
            raise ValueError("The scale parameters must be greater than zero.")

        self.low = low
        self.target = target
        self.high = high
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.tol = tol
        self.missing = None
        self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        if missing is None:
            missing = self.missing

        out = np.full(len(newdata), np.nan)
        out[(newdata < self.low) | (newdata > self.high)] = 0
        mask_low = (newdata <= self.target) & (newdata >= self.low)
        out[mask_low] = ((newdata[mask_low] - self.low) / (self.target - self.low)) ** self.low_scale
        mask_high = (newdata <= self.high) & (newdata >= self.target)
        out[mask_high] = ((newdata[mask_high] - self.high) / (self.target - self.high)) ** self.high_scale
        out[np.isnan(out)] = missing
        if self.tol is not None:
            out[out == 0] = self.tol
        return out


class DArb:
    def __init__(self, x, d, tol=None):
        if any(d > 1) or any(d < 0):
            raise ValueError("The desirability values must be 0 <= d <= 1.")
        if len(x) != len(d):
            raise ValueError("x and d must have the same length.")
        if len(x) < 2 or len(d) < 2:
            raise ValueError("x and d must have at least two values.")

        self.x = np.array(x)
        self.d = np.array(d)
        self.tol = tol
        self.missing = None
        self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(min(self.x), max(self.x), 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        if missing is None:
            missing = self.missing

        out = np.full(len(newdata), np.nan)
        out[newdata < min(self.x)] = self.d[0]
        out[newdata > max(self.x)] = self.d[-1]

        in_between = (newdata >= min(self.x)) & (newdata <= max(self.x))
        if np.any(in_between):
            out[in_between] = np.interp(newdata[in_between], self.x, self.d)

        out[np.isnan(out)] = missing
        if self.tol is not None:
            out[out == 0] = self.tol
        return out


class DBox:
    def __init__(self, low, high, tol=None):
        if low >= high:
            raise ValueError("The low value must be less than the high value.")

        self.low = low
        self.high = high
        self.tol = tol
        self.missing = None
        self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        if missing is None:
            missing = self.missing

        out = np.full(len(newdata), np.nan)
        out[(newdata < self.low) | (newdata > self.high)] = 0
        out[(newdata >= self.low) & (newdata <= self.high)] = 1
        out[np.isnan(out)] = missing
        if self.tol is not None:
            out[out == 0] = self.tol
        return out


class DCategorical:
    def __init__(self, values, tol=None):
        if len(values) < 2:
            raise ValueError("'values' should have at least two values.")
        if not all(isinstance(k, str) for k in values.keys()):
            raise ValueError("'values' should be a named dictionary.")

        self.values = values
        self.tol = tol
        self.missing = None
        self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        return np.mean(list(self.values.values()))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        if missing is None:
            missing = self.missing

        out = np.full(len(newdata), np.nan)
        for i, val in enumerate(newdata):
            if val in self.values:
                out[i] = self.values[val]
            else:
                raise ValueError(f"Value '{val}' not in allowed values: {list(self.values.keys())}")

        out[np.isnan(out)] = missing
        if self.tol is not None:
            out[out == 0] = self.tol
        return out


class DOverall:
    def __init__(self, *d_objs):
        valid_classes = (DMax, DMin, DTarget, DArb, DBox, DCategorical)
        if not all(isinstance(self, valid_classes) for obj in d_objs):
            raise ValueError("All objects must be instances of valid desirability classes.")

        self.d_objs = d_objs

    def predict(self, newdata=None, all=False):
        if newdata is None:
            newdata = np.full((1, len(self.d)), np.nan)

        if isinstance(newdata, np.ndarray) and newdata.ndim == 1:
            newdata = newdata.reshape(1, -1)

        if len(self.d) != newdata.shape[1]:
            raise ValueError("The number of columns in newdata must match the number of desirability functions.")

        out = np.full((newdata.shape[0], newdata.shape[1] + int(all)), np.nan)
        for i, d_obj in enumerate(self.d):
            out[:, i] = self.predict(newdata[:, i])

        if all:
            out[:, -1] = np.prod(out[:, :-1], axis=1) ** (1 / len(self.d))
        else:
            out = np.prod(out, axis=1) ** (1 / len(self.d))

        return out


class DesirabilityPrinter:
    @staticmethod
    def print_dBox(obj, digits=3, print_call=True):
        print("Box-like desirability function")
        if print_call and hasattr(obj, "call"):
            print(f"\nCall: {obj.call}\n")
        print(f"Non-informative value: {round(obj.missing, digits)}")
        if hasattr(obj, "tol") and obj.tol is not None:
            print(f"Tolerance: {round(obj.tol, digits)}")

    @staticmethod
    def print_dMax(obj, digits=3, print_call=True):
        print("Larger-is-better desirability function")
        if print_call and hasattr(obj, "call"):
            print(f"\nCall: {obj.call}\n")
        print(f"Non-informative value: {round(obj.missing, digits)}")
        if hasattr(obj, "tol") and obj.tol is not None:
            print(f"Tolerance: {round(obj.tol, digits)}")

    @staticmethod
    def print_dMin(obj, digits=3, print_call=True):
        print("Smaller-is-better desirability function")
        if print_call and hasattr(obj, "call"):
            print(f"\nCall: {obj.call}\n")
        print(f"Non-informative value: {round(obj.missing, digits)}")
        if hasattr(obj, "tol") and obj.tol is not None:
            print(f"Tolerance: {round(obj.tol, digits)}")

    @staticmethod
    def print_dTarget(obj, digits=3, print_call=True):
        print("Target-is-best desirability function")
        if print_call and hasattr(obj, "call"):
            print(f"\nCall: {obj.call}\n")
        print(f"Non-informative value: {round(obj.missing, digits)}")
        if hasattr(obj, "tol") and obj.tol is not None:
            print(f"Tolerance: {round(obj.tol, digits)}")

    @staticmethod
    def print_dArb(obj, digits=3, print_call=True):
        print("Arbitrary desirability function")
        if print_call and hasattr(obj, "call"):
            print(f"\nCall: {obj.call}\n")
        print(f"Non-informative value: {round(obj.missing, digits)}")
        if hasattr(obj, "tol") and obj.tol is not None:
            print(f"Tolerance: {round(obj.tol, digits)}")

    @staticmethod
    def print_dCategorical(obj, digits=3, print_call=True):
        print("Desirability function for categorical data")
        if print_call and hasattr(obj, "call"):
            print(f"\nCall: {obj.call}\n")
        print(f"Non-informative value: {round(obj.missing, digits)}")
        if hasattr(obj, "tol") and obj.tol is not None:
            print(f"Tolerance: {round(obj.tol, digits)}")

    @staticmethod
    def print_dOverall(obj, digits=3, print_call=True):
        print("Combined desirability function")
        if print_call and hasattr(obj, "call"):
            print(f"\nCall: {obj.call}\n")
        for i, d_obj in enumerate(obj.d, start=1):
            print("----")
            DesirabilityPrinter.print_dBox(d_obj, digits=digits, print_call=False)


def extend_range(values, factor=0.05):
    """Extend the range of values by a given factor."""
    range_span = max(values) - min(values)
    return [min(values) - factor * range_span, max(values) + factor * range_span]


def plot_dBox(obj, add=False, non_inform=True, **kwargs):
    x_range = extend_range([obj.low, obj.high])
    if not add:
        plt.plot([], [])  # Create an empty plot
        plt.xlim(x_range)
        plt.ylim(0, 1)
        plt.xlabel("Input")
        plt.ylabel("Desirability")
    plt.hlines(0, x_range[0], obj.low, **kwargs)
    plt.hlines(0, obj.high, x_range[1], **kwargs)
    plt.vlines(obj.low, 0, 1, **kwargs)
    plt.vlines(obj.high, 0, 1, **kwargs)
    plt.hlines(1, obj.low, obj.high, **kwargs)
    if non_inform:
        plt.axhline(y=obj.missing, linestyle="--", **kwargs)
    plt.show()


def plot_dMin(obj, add=False, non_inform=True, **kwargs):
    x_range = extend_range([obj.low, obj.high])
    if not add:
        plt.plot([], [])  # Create an empty plot
        plt.xlim(x_range)
        plt.ylim(0, 1)
        plt.xlabel("Input")
        plt.ylabel("Desirability")
    plt.hlines(1, x_range[0], obj.low, **kwargs)
    plt.hlines(0, obj.high, x_range[1], **kwargs)
    input_values = np.linspace(obj.low, obj.high, 100)
    output_values = obj.predict(input_values)
    plt.plot(input_values, output_values, **kwargs)
    if non_inform:
        plt.axhline(y=obj.missing, linestyle="--", **kwargs)
    plt.show()


def plot_dTarget(obj, add=False, non_inform=True, **kwargs):
    x_range = extend_range([obj.low, obj.high])
    if not add:
        plt.plot([], [])  # Create an empty plot
        plt.xlim(x_range)
        plt.ylim(0, 1)
        plt.xlabel("Input")
        plt.ylabel("Desirability")
    plt.hlines(0, x_range[0], obj.low, **kwargs)
    plt.hlines(0, obj.high, x_range[1], **kwargs)
    input_values = np.linspace(obj.low, obj.high, 100)
    output_values = obj.predict(input_values)
    plt.plot(input_values, output_values, **kwargs)
    if non_inform:
        plt.axhline(y=obj.missing, linestyle="--", **kwargs)
    plt.show()


def plot_dCategorical(obj, non_inform=True, **kwargs):
    plt.bar(range(len(obj.values)), list(obj.values.values()), tick_label=list(obj.values.keys()), **kwargs)
    plt.ylabel("Desirability")
    if non_inform:
        plt.axhline(y=obj.missing, linestyle="--", **kwargs)
    plt.show()


def plot_dArb(obj, add=False, non_inform=True, **kwargs):
    x_range = extend_range(obj.x)
    if not add:
        plt.plot([], [])  # Create an empty plot
        plt.xlim(x_range)
        plt.ylim(0, 1)
        plt.xlabel("Input")
        plt.ylabel("Desirability")
    input_values = np.linspace(x_range[0], x_range[1], 100)
    output_values = obj.predict(input_values)
    plt.plot(input_values, output_values, **kwargs)
    if non_inform:
        plt.axhline(y=obj.missing, linestyle="--", **kwargs)
    plt.show()


def plot_dMax(obj, add=False, non_inform=True, **kwargs):
    x_range = extend_range([obj.low, obj.high])
    if not add:
        plt.plot([], [])  # Create an empty plot
        plt.xlim(x_range)
        plt.ylim(0, 1)
        plt.xlabel("Input")
        plt.ylabel("Desirability")
    plt.hlines(0, x_range[0], obj.low, **kwargs)
    plt.hlines(1, obj.high, x_range[1], **kwargs)
    input_values = np.linspace(obj.low, obj.high, 100)
    output_values = obj.predict(input_values)
    plt.plot(input_values, output_values, **kwargs)
    if non_inform:
        plt.axhline(y=obj.missing, linestyle="--", **kwargs)
    plt.show()
