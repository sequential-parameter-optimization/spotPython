import numpy as np
import matplotlib.pyplot as plt


class DesirabilityBase:
    """
    Base class for all desirability functions. Provides a method to print class attributes.
    """

    def print_class_attributes(self, indent=0):
        """
        Prints the attributes of the class object in a generic and recursive manner.

        Args:
            indent (int): The indentation level for nested objects.
        """
        # Print the class name of the current object
        print("\n" + " " * indent + f"Class: {type(self).__name__}")

        # Get the attributes of the object as a dictionary
        attributes = vars(self)
        for attr, value in attributes.items():
            if isinstance(value, DesirabilityBase):  # Check if the attribute is another desirability object
                print(" " * indent + f"{attr}:")
                value.print_class_attributes(indent=indent + 2)  # Recursive call with increased indentation
            elif isinstance(value, (list, tuple)) and all(isinstance(v, DesirabilityBase) for v in value):
                print(" " * indent + f"{attr}: [")
                for v in value:
                    v.print_class_attributes(indent=indent + 2)  # Recursive call for each object in the list/tuple
                print(" " * indent + "]")
            else:
                print(" " * indent + f"{attr}: {value}")


class DMax(DesirabilityBase):
    def __init__(self, low, high, scale=1, tol=None, missing=None):
        if low >= high:
            raise ValueError("The low value must be less than the high value.")
        if scale <= 0:
            raise ValueError("The scale parameter must be greater than zero.")

        self.low = low
        self.high = high
        self.scale = scale
        self.tol = tol
        self.missing = missing
        if self.missing is None:
            self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        elif isinstance(newdata, (int, float)):  # Handle single float or int input
            newdata = np.array([newdata])
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

    def plot(self, add=False, non_inform=True, **kwargs):
        x_range = extend_range([self.low, self.high])
        if not add:
            plt.plot([], [])  # Create an empty plot
            plt.xlim(x_range)
            plt.ylim(0, 1)
            plt.xlabel("Input")
            plt.ylabel("Desirability")
        plt.hlines(0, x_range[0], self.low, **kwargs)
        plt.hlines(1, self.high, x_range[1], **kwargs)
        input_values = np.linspace(self.low, self.high, 100)
        output_values = self.predict(input_values)
        plt.plot(input_values, output_values, **kwargs)
        if non_inform:
            plt.axhline(y=self.missing, linestyle="--", **kwargs)
        plt.show()


class DMin(DesirabilityBase):
    def __init__(self, low, high, scale=1, tol=None, missing=None):
        if low >= high:
            raise ValueError("The low value must be less than the high value.")
        if scale <= 0:
            raise ValueError("The scale parameter must be greater than zero.")

        self.low = low
        self.high = high
        self.scale = scale
        self.tol = tol
        self.missing = missing
        if self.missing is None:
            self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        elif isinstance(newdata, (int, float)):  # Handle single float or int input
            newdata = np.array([newdata])
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

    def plot(self, add=False, non_inform=True, **kwargs):
        x_range = extend_range([self.low, self.high])
        if not add:
            plt.plot([], [])  # Create an empty plot
            plt.xlim(x_range)
            plt.ylim(0, 1)
            plt.xlabel("Input")
            plt.ylabel("Desirability")
        plt.hlines(1, x_range[0], self.low, **kwargs)
        plt.hlines(0, self.high, x_range[1], **kwargs)
        input_values = np.linspace(self.low, self.high, 100)
        output_values = self.predict(input_values)
        plt.plot(input_values, output_values, **kwargs)
        if non_inform:
            plt.axhline(y=self.missing, linestyle="--", **kwargs)
        plt.show()


class DTarget(DesirabilityBase):
    def __init__(self, low, target, high, low_scale=1, high_scale=1, tol=None, missing=None):
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
        self.missing = missing
        if self.missing is None:
            self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        elif isinstance(newdata, (int, float)):  # Handle single float or int input
            newdata = np.array([newdata])
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

    def plot(self, add=False, non_inform=True, **kwargs):
        x_range = extend_range([self.low, self.high])
        if not add:
            plt.plot([], [])  # Create an empty plot
            plt.xlim(x_range)
            plt.ylim(0, 1)
            plt.xlabel("Input")
            plt.ylabel("Desirability")
        plt.hlines(0, x_range[0], self.low, **kwargs)
        plt.hlines(0, self.high, x_range[1], **kwargs)
        input_values = np.linspace(self.low, self.high, 100)
        output_values = self.predict(input_values)
        plt.plot(input_values, output_values, **kwargs)
        if non_inform:
            plt.axhline(y=self.missing, linestyle="--", **kwargs)
        plt.show()


class DArb(DesirabilityBase):
    def __init__(self, x, d, tol=None, missing=None):
        if any(d > 1) or any(d < 0):
            raise ValueError("The desirability values must be 0 <= d <= 1.")
        if len(x) != len(d):
            raise ValueError("x and d must have the same length.")
        if len(x) < 2 or len(d) < 2:
            raise ValueError("x and d must have at least two values.")

        self.x = np.array(x)
        self.d = np.array(d)
        self.tol = tol
        self.missing = missing
        if self.missing is None:
            self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(min(self.x), max(self.x), 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        elif isinstance(newdata, (int, float)):  # Handle single float or int input
            newdata = np.array([newdata])
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

    def plot(self, add=False, non_inform=True, **kwargs):
        x_range = extend_range(self.x)
        if not add:
            plt.plot([], [])  # Create an empty plot
            plt.xlim(x_range)
            plt.ylim(0, 1)
            plt.xlabel("Input")
            plt.ylabel("Desirability")
        input_values = np.linspace(x_range[0], x_range[1], 100)
        output_values = self.predict(input_values)
        plt.plot(input_values, output_values, **kwargs)
        if non_inform:
            plt.axhline(y=self.missing, linestyle="--", **kwargs)
        plt.show()


class DBox(DesirabilityBase):
    def __init__(self, low, high, tol=None, missing=None):
        if low >= high:
            raise ValueError("The low value must be less than the high value.")

        self.low = low
        self.high = high
        self.tol = tol
        self.missing = missing
        if self.missing is None:
            self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        elif isinstance(newdata, (int, float)):  # Handle single float or int input
            newdata = np.array([newdata])
        if missing is None:
            missing = self.missing

        out = np.full(len(newdata), np.nan)
        out[(newdata < self.low) | (newdata > self.high)] = 0
        out[(newdata >= self.low) & (newdata <= self.high)] = 1
        out[np.isnan(out)] = missing
        if self.tol is not None:
            out[out == 0] = self.tol
        return out

    def plot(self, add=False, non_inform=True, **kwargs):
        x_range = extend_range([self.low, self.high])
        if not add:
            plt.plot([], [])  # Create an empty plot
            plt.xlim(x_range)
            plt.ylim(0, 1)
            plt.xlabel("Input")
            plt.ylabel("Desirability")
        plt.hlines(0, x_range[0], self.low, **kwargs)
        plt.hlines(0, self.high, x_range[1], **kwargs)
        plt.vlines(self.low, 0, 1, **kwargs)
        plt.vlines(self.high, 0, 1, **kwargs)
        plt.hlines(1, self.low, self.high, **kwargs)
        if non_inform:
            plt.axhline(y=self.missing, linestyle="--", **kwargs)
        plt.show()


class DCategorical(DesirabilityBase):
    def __init__(self, values, tol=None, missing=None):
        if len(values) < 2:
            raise ValueError("'values' should have at least two values.")
        if not all(isinstance(k, str) for k in values.keys()):
            raise ValueError("'values' should be a named dictionary.")

        self.values = values
        self.tol = tol
        self.missing = missing
        if self.missing is None:
            self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        return np.mean(list(self.values.values()))

    def predict(self, newdata=None, missing=None):
        if newdata is None:
            newdata = np.array([])
        elif isinstance(newdata, (int, float)):  # Handle single float or int input
            newdata = np.array([newdata])
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

    def plot(self, non_inform=True, **kwargs):
        plt.bar(range(len(self.values)), list(self.values.values()), tick_label=list(self.values.keys()), **kwargs)
        plt.ylabel("Desirability")
        if non_inform:
            plt.axhline(y=self.missing, linestyle="--", **kwargs)
        plt.show()


class DOverall(DesirabilityBase):
    def __init__(self, *d_objs):
        """
        Combines multiple desirability objects into an overall desirability function.

        Args:
            *d_objs: Instances of desirability classes (e.g., DMax, DTarget, etc.).
        """
        valid_classes = (DMax, DMin, DTarget, DArb, DBox, DCategorical)
        # print the instanaces of desirability classes
        print(f"d_objs: {d_objs}")
        for obj in d_objs:
            print(f"obj: {obj}")
            print(f"isinstance(obj, valid_classes): {isinstance(obj, valid_classes)}")

        if not all(isinstance(obj, valid_classes) for obj in d_objs):
            raise ValueError("All objects must be instances of valid desirability classes.")

        self.d_objs = d_objs  # Store the desirability objects

    def predict(self, newdata, all=False):
        """
        Predicts the overall desirability based on the individual desirability objects.

        Args:
            newdata (list or numpy array): A list or array of predicted outcomes, one for each desirability object.
            all (bool): Whether to return individual desirabilities along with the overall desirability.

        Returns:
            float or tuple: The overall desirability score, or a tuple of individual and overall desirabilities if `all=True`.
        """

        # # Compute individual desirabilities
        # individual_desirabilities = [self.predict(np.array([value]))[0] for obj, value in zip(self.d_objs, newdata)]
        # Updated: Compute individual desirabilities
        # Ensure newdata is a NumPy array
        newdata = np.array(newdata)

        # BEGIN Modified in 0.27.2: Allow 1D array as input
        # Validate the shape of newdata
        if newdata.ndim == 1:
            newdata = newdata.reshape(1, -1)  # Reshape 1D array to 2D array with one row

        if newdata.shape[1] != len(self.d_objs):
            print(f"newdata.shape: {newdata.shape}")
            print(f"len(self.d_objs): {len(self.d_objs)}")
            raise ValueError("The number of columns in newdata must match the number of desirability objects.")
        # END Modify

        # Compute individual desirabilities
        individual_desirabilities = [obj.predict(newdata[:, i]) for i, obj in enumerate(self.d_objs)]

        # Compute the geometric mean of the individual desirabilities
        overall_desirability = np.prod(individual_desirabilities, axis=0) ** (1 / len(individual_desirabilities))

        if all:
            return individual_desirabilities, overall_desirability
        return overall_desirability


class DesirabilityPrinter:
    @staticmethod
    def print_dBox(self, digits=3, print_call=True):
        print("Box-like desirability function")
        if print_call and hasattr(self, "call"):
            print(f"\nCall: {self.call}\n")
        print(f"Non-informative value: {round(self.missing, digits)}")
        if hasattr(self, "tol") and self.tol is not None:
            print(f"Tolerance: {round(self.tol, digits)}")

    @staticmethod
    def print_dMax(self, digits=3, print_call=True):
        print("Larger-is-better desirability function")
        if print_call and hasattr(self, "call"):
            print(f"\nCall: {self.call}\n")
        print(f"Non-informative value: {round(self.missing, digits)}")
        if hasattr(self, "tol") and self.tol is not None:
            print(f"Tolerance: {round(self.tol, digits)}")

    @staticmethod
    def print_dMin(self, digits=3, print_call=True):
        print("Smaller-is-better desirability function")
        if print_call and hasattr(self, "call"):
            print(f"\nCall: {self.call}\n")
        print(f"Non-informative value: {round(self.missing, digits)}")
        if hasattr(self, "tol") and self.tol is not None:
            print(f"Tolerance: {round(self.tol, digits)}")

    @staticmethod
    def print_dTarget(self, digits=3, print_call=True):
        print("Target-is-best desirability function")
        if print_call and hasattr(self, "call"):
            print(f"\nCall: {self.call}\n")
        print(f"Non-informative value: {round(self.missing, digits)}")
        if hasattr(self, "tol") and self.tol is not None:
            print(f"Tolerance: {round(self.tol, digits)}")

    @staticmethod
    def print_dArb(self, digits=3, print_call=True):
        print("Arbitrary desirability function")
        if print_call and hasattr(self, "call"):
            print(f"\nCall: {self.call}\n")
        print(f"Non-informative value: {round(self.missing, digits)}")
        if hasattr(self, "tol") and self.tol is not None:
            print(f"Tolerance: {round(self.tol, digits)}")

    @staticmethod
    def print_dCategorical(self, digits=3, print_call=True):
        print("Desirability function for categorical data")
        if print_call and hasattr(self, "call"):
            print(f"\nCall: {self.call}\n")
        print(f"Non-informative value: {round(self.missing, digits)}")
        if hasattr(self, "tol") and self.tol is not None:
            print(f"Tolerance: {round(self.tol, digits)}")

    @staticmethod
    def print_dOverall(self, digits=3, print_call=True):
        print("Combined desirability function")
        if print_call and hasattr(self, "call"):
            print(f"\nCall: {self.call}\n")
        for i, d_obj in enumerate(self.d, start=1):
            print("----")
            DesirabilityPrinter.print_dBox(d_obj, digits=digits, print_call=False)


def extend_range(values, factor=0.05):
    """Extend the range of values by a given factor."""
    range_span = max(values) - min(values)
    return [min(values) - factor * range_span, max(values) + factor * range_span]


def conversion_pred(x):
    """
    Predicts the percent conversion based on the input vector x.

    Args:
        x (list or numpy array): A vector of three input values [x1, x2, x3].

    Returns:
        float: The predicted percent conversion.
    """
    return 81.09 + 1.0284 * x[0] + 4.043 * x[1] + 6.2037 * x[2] - 1.8366 * x[0] ** 2 + 2.9382 * x[1] ** 2 - 5.1915 * x[2] ** 2 + 2.2150 * x[0] * x[1] + 11.375 * x[0] * x[2] - 3.875 * x[1] * x[2]


def activity_pred(x):
    """
    Predicts the thermal activity based on the input vector x.

    Args:
        x (list or numpy array): A vector of three input values [x1, x2, x3].

    Returns:
        float: The predicted thermal activity.
    """
    return 59.85 + 3.583 * x[0] + 0.2546 * x[1] + 2.2298 * x[2] + 0.83479 * x[0] ** 2 + 0.07484 * x[1] ** 2 + 0.05716 * x[2] ** 2 - 0.3875 * x[0] * x[1] - 0.375 * x[0] * x[2] + 0.3125 * x[1] * x[2]
