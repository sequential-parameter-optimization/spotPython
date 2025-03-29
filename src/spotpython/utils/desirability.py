import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict


class DesirabilityBase:
    """Base class for all desirability functions.

    Provides a method to print class attributes and extend the range of values.

    Methods:
        print_class_attributes(indent=0):
            Prints the attributes of the class object in a generic and recursive manner.

        extend_range(values, factor=0.05):
            Extends the range of values by a given factor.

    References:
        Many thanks to Max Kuhn for his implementation of the 'desirability' package in R.
        This class is based on the 'desirability' package in R, see:
        https://cran.r-project.org/package=desirability
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

    def extend_range(self, values, factor=0.05):
        """Extend the range of values by a given factor."""
        range_span = max(values) - min(values)
        return [min(values) - factor * range_span, max(values) + factor * range_span]


class DMax(DesirabilityBase):
    """
    Implements a desirability function for maximization.

    The desirability function assigns a value of 0 for inputs below the `low` threshold,
    a value of 1 for inputs above the `high` threshold, and scales the desirability
    between 0 and 1 for inputs within the range `[low, high]` using a specified scale factor.

    Attributes:
        low (float): The lower threshold for the desirability function.
        high (float): The upper threshold for the desirability function.
        scale (float): The scaling factor for the desirability function. Must be greater than 0.
        tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
        missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

    Methods:
        predict(newdata=None, missing=None):
            Predicts the desirability values for the given input data.

        plot(add=False, non_inform=True, **kwargs):
            Plots the desirability function.

    References:
        Many thanks to Max Kuhn for his implementation of the 'desirability' package in R.
        This class is based on the 'desirability' package in R, see:
        https://cran.r-project.org/package=desirability

    Examples:
        >>> from spotpython.utils.desirability import DMax
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        # Create a DMax object
        >>> dmax = DMax(low=0, high=10, scale=1)

        # Predict desirability for a range of inputs
        >>> inputs = np.array([-5, 0, 5, 10, 15])
        >>> desirability = dmax.predict(inputs)
        >>> print(desirability)
        [0. 0. 0.5 1. 1.]

        # Plot the desirability function
        >>> dmax.plot()
    """

    def __init__(self, low, high, scale=1, tol=None, missing=None):
        """
        Initializes the DMax object.

        Args:
            low (float): The lower threshold for the desirability function.
            high (float): The upper threshold for the desirability function.
            scale (float): The scaling factor for the desirability function. Must be greater than 0.
            tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

        Raises:
            ValueError: If `low` is greater than or equal to `high`.
            ValueError: If `scale` is less than or equal to 0.
        """
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

    def _calculate_non_informative_value(self) -> float:
        """
        Calculates a non-informative value for missing inputs.

        Returns:
            float: The mean desirability value over the range `[low, high]`.
        """
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None) -> np.ndarray:
        """
        Predicts the desirability values for the given input data.

        Args:
            newdata (array-like, optional): The input data for which to compute desirability values.
                If None, an empty array is used. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to the object's `missing` attribute.

        Returns:
            numpy.ndarray: The desirability values for the input data.

        Examples:
            >>> from spotpython.utils.desirability import DMax
            >>> import numpy as np
            >>> dmax = DMax(low=0, high=10, scale=1)
            >>> inputs = np.array([-5, 0, 5, 10, 15])
            >>> desirability = dmax.predict(inputs)
            >>> print(desirability)
            [0. 0. 0.5 1. 1.]
        """
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

    def plot(self, add: bool = False, non_inform: bool = True, **kwargs: Dict[str, Any]) -> None:
        """
        Plots the desirability function.

        Args:
            add (bool, optional): Whether to add the plot to an existing figure. Defaults to False.
            non_inform (bool, optional): Whether to display the non-informative value as a dashed line. Defaults to True.
            **kwargs (Dict[str, Any]): Additional keyword arguments for the plot.

        Examples:
            >>> from spotpython.utils.desirability import DMax
            >>> dmax = DMax(low=0, high=10, scale=1)
            >>> dmax.plot()
        """
        x_range = self.extend_range([self.low, self.high])
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
    """
    Implements a desirability function for minimization.

    The desirability function assigns a value of 1 for inputs below the `low` threshold,
    a value of 0 for inputs above the `high` threshold, and scales the desirability
    between 1 and 0 for inputs within the range `[low, high]` using a specified scale factor.

    Attributes:
        low (float): The lower threshold for the desirability function.
        high (float): The upper threshold for the desirability function.
        scale (float): The scaling factor for the desirability function. Must be greater than 0.
        tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
        missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

    Methods:
        predict(newdata=None, missing=None):
            Predicts the desirability values for the given input data.

        plot(add=False, non_inform=True, **kwargs):
            Plots the desirability function.

    References:
        Many thanks to Max Kuhn for his implementation of the 'desirability' package in R.
        This class is based on the 'desirability' package in R, see:
        https://cran.r-project.org/package=desirability

    Examples:
        >>> from spotpython.utils.desirability import DMin
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        # Create a DMin object
        >>> dmin = DMin(low=0, high=10, scale=1)

        # Predict desirability for a range of inputs
        >>> inputs = np.array([-5, 0, 5, 10, 15])
        >>> desirability = dmin.predict(inputs)
        >>> print(desirability)
        [1. 1. 0.5 0. 0.]

        # Plot the desirability function
        >>> dmin.plot()
    """

    def __init__(self, low, high, scale=1, tol=None, missing=None):
        """
        Initializes the DMin object.

        Args:
            low (float): The lower threshold for the desirability function.
            high (float): The upper threshold for the desirability function.
            scale (float): The scaling factor for the desirability function. Must be greater than 0.
            tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

        Raises:
            ValueError: If `low` is greater than or equal to `high`.
            ValueError: If `scale` is less than or equal to 0.
        """
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
        """
        Calculates a non-informative value for missing inputs.

        Returns:
            float: The mean desirability value over the range `[low, high]`.
        """
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        """
        Predicts the desirability values for the given input data.

        Args:
            newdata (array-like, optional): The input data for which to compute desirability values.
                If None, an empty array is used. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to the object's `missing` attribute.

        Returns:
            (numpy.ndarray): The desirability values for the input data.

        Examples:
            >>> from spotpython.utils.desirability import DMin
            >>> dmin = DMin(low=0, high=10, scale=1)
            >>> inputs = np.array([-5, 0, 5, 10, 15])
            >>> desirability = dmin.predict(inputs)
            >>> print(desirability)
            [1. 1. 0.5 0. 0.]
        """
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

    def plot(self, add: bool = False, non_inform: bool = True, **kwargs: Dict[str, Any]) -> None:
        """
        Plots the desirability function.

        Args:
            add (bool, optional): Whether to add the plot to an existing figure. Defaults to False.
            non_inform (bool, optional): Whether to display the non-informative value as a dashed line. Defaults to True.
            **kwargs: Additional keyword arguments for the plot.

        Examples:
            >>> from spotpython.utils.desirability import DMin
            >>> dmin = DMin(low=0, high=10, scale=1)
            >>> dmin.plot()
        """
        x_range = self.extend_range([self.low, self.high])
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
    """
    Implements a desirability function for target optimization.

    The desirability function assigns a value of 0 for inputs outside the range `[low, high]`,
    scales the desirability between 0 and 1 for inputs within `[low, target]` using `low_scale`,
    and scales the desirability between 1 and 0 for inputs within `[target, high]` using `high_scale`.

    Attributes:
        low (float): The lower threshold for the desirability function.
        target (float): The target value for the desirability function.
        high (float): The upper threshold for the desirability function.
        low_scale (float): The scaling factor for the desirability function below the target. Must be greater than 0.
        high_scale (float): The scaling factor for the desirability function above the target. Must be greater than 0.
        tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
        missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

    Methods:
        predict(newdata=None, missing=None):
            Predicts the desirability values for the given input data.

        plot(add=False, non_inform=True, **kwargs):
            Plots the desirability function.

    References:
        Many thanks to Max Kuhn for his implementation of the 'desirability' package in R.
        This class is based on the 'desirability' package in R, see:
        https://cran.r-project.org/package=desirability

    Examples:
        >>> from spotpython.utils.desirability import DTarget
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        # Create a DTarget object
        >>> dtarget = DTarget(low=0, target=5, high=10, low_scale=1, high_scale=1)
        # Predict desirability for a range of inputs
        >>> inputs = np.array([-5, 0, 2.5, 5, 7.5, 10, 15])
        >>> desirability = dtarget.predict(inputs)
        >>> print(desirability)
        [0.   0.   0.5  1.   0.5  0.   0.  ]
        # Plot the desirability function
        >>> dtarget.plot()
    """

    def __init__(self, low, target, high, low_scale=1, high_scale=1, tol=None, missing=None):
        """
        Initializes the DTarget object.

        Args:
            low (float): The lower threshold for the desirability function.
            target (float): The target value for the desirability function.
            high (float): The upper threshold for the desirability function.
            low_scale (float): The scaling factor for the desirability function below the target. Must be greater than 0.
            high_scale (float): The scaling factor for the desirability function above the target. Must be greater than 0.
            tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

        Raises:
            ValueError: If `low` is greater than or equal to `high`.
            ValueError: If `low` is greater than or equal to `target`.
            ValueError: If `target` is greater than or equal to `high`.
            ValueError: If `low_scale` or `high_scale` is less than or equal to 0.
        """
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
        """
        Calculates a non-informative value for missing inputs.

        Returns:
            float: The mean desirability value over the range `[low, high]`.
        """
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        """
        Predicts the desirability values for the given input data.

        Args:
            newdata (array-like, optional): The input data for which to compute desirability values.
                If None, an empty array is used. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to the object's `missing` attribute.

        Returns:
            (numpy.ndarray): The desirability values for the input data.

        Examples:
            >>> from spotpython.utils.desirability import DTarget
            >>> import numpy as np
            # Create a DTarget object
            >>> dtarget = DTarget(low=0, target=5, high=10, low_scale=1, high_scale=1)
            >>> inputs = np.array([-5, 0, 2.5, 5, 7.5, 10, 15])
            >>> desirability = dtarget.predict(inputs)
            >>> print(desirability)
            [0.   0.   0.5  1.   0.5  0.   0.  ]
        """
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

    def plot(self, add: bool = False, non_inform: bool = True, **kwargs: Dict[str, Any]) -> None:
        """
        Plots the desirability function.

        Args:
            add (bool, optional): Whether to add the plot to an existing figure. Defaults to False.
            non_inform (bool, optional): Whether to display the non-informative value as a dashed line. Defaults to True.
            **kwargs: Additional keyword arguments for the plot.

        Examples:
            >>> from spotpython.utils.desirability import DTarget
            >>> dtarget = DTarget(low=0, target=5, high=10, low_scale=1, high_scale=1)
            >>> dtarget.plot()
        """
        x_range = self.extend_range([self.low, self.high])
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
    """
    Implements an arbitrary desirability function.

    This class allows users to define a custom desirability function by specifying
    input values (`x`) and their corresponding desirability values (`d`).

    Attributes:
        x (numpy.ndarray): The input values for the desirability function.
        d (numpy.ndarray): The desirability values corresponding to the input values.
        tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
        missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

    Methods:
        predict(newdata=None, missing=None):
            Predicts the desirability values for the given input data.

        plot(add=False, non_inform=True, **kwargs):
            Plots the desirability function.

    References:
        Many thanks to Max Kuhn for his implementation of the 'desirability' package in R.
        This class is based on the 'desirability' package in R, see:
        https://cran.r-project.org/package=desirability

    Examples:
        >>> from spotpython.utils.desirability import DArb
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        # Define input and desirability values
        >>> x = [-5, 0, 5, 10]
        >>> d = [0, 0.5, 1, 0.2]
        # Create a DArb object
        >>> darb = DArb(x, d)
        # Predict desirability for a range of inputs
        >>> inputs = np.array([-10, -5, 0, 5, 10, 15])
        >>> desirability = darb.predict(inputs)
        >>> print(desirability)
        [0.  0.  0.5 1.  0.2 0.2]
        # Plot the desirability function
        >>> darb.plot()
    """

    def __init__(self, x, d, tol=None, missing=None):
        """
        Initializes the DArb object.

        Args:
            x (list or numpy.ndarray): The input values for the desirability function.
            d (list or numpy.ndarray): The desirability values corresponding to the input values.
            tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

        Raises:
            ValueError: If any desirability value is not in the range [0, 1].
            ValueError: If `x` and `d` do not have the same length.
            ValueError: If `x` or `d` has fewer than two values.
        """
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
        """
        Calculates a non-informative value for missing inputs.

        Returns:
            float: The mean desirability value over the range `[min(x), max(x)]`.
        """
        test_seq = np.linspace(min(self.x), max(self.x), 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        """
        Predicts the desirability values for the given input data.

        Args:
            newdata (array-like, optional): The input data for which to compute desirability values.
                If None, an empty array is used. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to the object's `missing` attribute.

        Returns:
            (numpy.ndarray): The desirability values for the input data.

        Examples:
            >>> from spotpython.utils.desirability import DArb
            >>> import numpy as np
            # Define input and desirability values
            >>> x = [-5, 0, 5, 10]
            >>> d = [0, 0.5, 1, 0.2]
            >>> darb = DArb(x, d)
            >>> inputs = np.array([-10, -5, 0, 5, 10, 15])
            >>> desirability = darb.predict(inputs)
            >>> print(desirability)
            [0.  0.  0.5 1.  0.2 0.2]
        """
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

    def plot(self, add: bool = False, non_inform: bool = True, **kwargs: Dict[str, Any]) -> None:
        """
        Plots the desirability function.

        Args:
            add (bool, optional): Whether to add the plot to an existing figure. Defaults to False.
            non_inform (bool, optional): Whether to display the non-informative value as a dashed line. Defaults to True.
            **kwargs: Additional keyword arguments for the plot.

        Examples:
            >>> from spotpython.utils.desirability import DArb
            >>> x = [-5, 0, 5, 10]
            >>> d = [0, 0.5, 1, 0.2]
            >>> darb = DArb(x, d)
            >>> darb.plot()
        """
        x_range = self.extend_range(self.x)
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
    """
    Implements a box-like desirability function.

    The desirability function assigns a value of 1 for inputs within the range `[low, high]`
    and a value of 0 for inputs outside this range.

    Attributes:
        low (float): The lower threshold for the desirability function.
        high (float): The upper threshold for the desirability function.
        tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
        missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

    Methods:
        predict(newdata=None, missing=None):
            Predicts the desirability values for the given input data.

        plot(add=False, non_inform=True, **kwargs):
            Plots the desirability function.

    References:
        Many thanks to Max Kuhn for his implementation of the 'desirability' package in R.
        This class is based on the 'desirability' package in R, see:
        https://cran.r-project.org/package=desirability

    Examples:
        >>> from spotpython.utils.desirability import DBox
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        # Create a DBox object
        >>> dbox = DBox(low=-1.682, high=1.682)
        # Predict desirability for a range of inputs
        >>> inputs = np.array([-3, -1.682, 0, 1.682, 3])
        >>> desirability = dbox.predict(inputs)
        >>> print(desirability)
        [0. 1. 1. 1. 0.]
        # Plot the desirability function
        >>> dbox.plot()
    """

    def __init__(self, low, high, tol=None, missing=None):
        """
        Initializes the DBox object.

        Args:
            low (float): The lower threshold for the desirability function.
            high (float): The upper threshold for the desirability function.
            tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

        Raises:
            ValueError: If `low` is greater than or equal to `high`.
        """
        if low >= high:
            raise ValueError("The low value must be less than the high value.")

        self.low = low
        self.high = high
        self.tol = tol
        self.missing = missing
        if self.missing is None:
            self.missing = self._calculate_non_informative_value()

    def _calculate_non_informative_value(self):
        """
        Calculates a non-informative value for missing inputs.

        Returns:
            (float): The mean desirability value over the range `[low, high]`.
        """
        test_seq = np.linspace(self.low, self.high, 100)
        return np.mean(self.predict(test_seq))

    def predict(self, newdata=None, missing=None):
        """
        Predicts the desirability values for the given input data.

        Args:
            newdata (array-like, optional): The input data for which to compute desirability values.
                If None, an empty array is used. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to the object's `missing` attribute.

        Returns:
            (numpy.ndarray): The desirability values for the input data.

        Examples:
            >>> from spotpython.utils.desirability import DBox
            >>> import numpy as np
            # Create a DBox object
            >>> dbox = DBox(low=-1.682, high=1.682)
            >>> inputs = np.array([-3, -1.682, 0, 1.682, 3])
            >>> desirability = dbox.predict(inputs)
            >>> print(desirability)
            [0. 1. 1. 1. 0.]
        """
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

    def plot(self, add: bool = False, non_inform: bool = True, **kwargs: Dict[str, Any]) -> None:
        """
        Plots the desirability function.

        Args:
            add (bool, optional): Whether to add the plot to an existing figure. Defaults to False.
            non_inform (bool, optional): Whether to display the non-informative value as a dashed line. Defaults to True.
            **kwargs: Additional keyword arguments for the plot.

        Examples:
            >>> from spotpython.utils.desirability import DBox
            >>> dbox = DBox(low=-1.682, high=1.682)
            >>> dbox.plot()
        """
        x_range = self.extend_range([self.low, self.high])
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
    """
    Implements a desirability function for categorical inputs.

    This class allows users to define desirability values for specific categorical inputs.

    Attributes:
        values (dict): A dictionary where keys are category names (strings) and values are desirability scores (floats).
        tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
        missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

    Methods:
        predict(newdata=None, missing=None):
            Predicts the desirability values for the given categorical input data.

        plot(non_inform=True, **kwargs):
            Plots the desirability function for the categorical inputs.

    References:
        Many thanks to Max Kuhn for his implementation of the 'desirability' package in R.
        This class is based on the 'desirability' package in R, see:
        https://cran.r-project.org/package=desirability

    Examples:
        >>> from spotpython.utils.desirability import DCategorical
        >>> import matplotlib.pyplot as plt
        # Define desirability values for categories
        >>> values = {"A": 0.1, "B": 0.9, "C": 0.5}
        # Create a DCategorical object
        >>> dcat = DCategorical(values)
        # Predict desirability for a list of categories
        >>> inputs = ["A", "B", "C", "D"]
        >>> desirability = dcat.predict(inputs)
        >>> print(desirability)
        [0.1 0.9 0.5 ValueError: Value 'D' not in allowed values: ['A', 'B', 'C']]
        # Plot the desirability function
        >>> dcat.plot()
    """

    def __init__(self, values, tol=None, missing=None):
        """
        Initializes the DCategorical object.

        Args:
            values (dict): A dictionary where keys are category names (strings) and values are desirability scores (floats).
            tol (float, optional): A tolerance value to replace desirability values of 0. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to a non-informative value.

        Raises:
            ValueError: If `values` has fewer than two entries.
            ValueError: If keys in `values` are not strings.
        """
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
        """
        Calculates a non-informative value for missing inputs.

        Returns:
            (float): The mean desirability value across all categories.
        """
        return np.mean(list(self.values.values()))

    def predict(self, newdata=None, missing=None):
        """
        Predicts the desirability values for the given categorical input data.

        Args:
            newdata (list or array-like, optional): A list or array of categorical inputs.
                If None, an empty array is used. Defaults to None.
            missing (float, optional): The value to use for missing inputs. Defaults to the object's `missing` attribute.

        Returns:
            (numpy.ndarray): The desirability values for the input data.

        Raises:
            ValueError: If a category in `newdata` is not in the allowed categories.

        Examples:
            >>> from spotpython.utils.desirability import DCategorical
            >>> values = {"A": 0.1, "B": 0.9, "C": 0.5}
            >>> dcat = DCategorical(values)
            >>> inputs = ["A", "B", "C"]
            >>> desirability = dcat.predict(inputs)
            >>> print(desirability)
            [0.1 0.9 0.5]
        """
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

    def plot(self, non_inform: bool = True, **kwargs: Dict[str, Any]) -> None:
        """
        Plots the desirability function for the categorical inputs.

        Args:
            non_inform (bool, optional): Whether to display the non-informative value as a dashed line. Defaults to True.
            **kwargs: Additional keyword arguments for the plot.

        Examples:
            >>> from spotpython.utils.desirability import DCategorical
            >>> values = {"A": 0.1, "B": 0.9, "C": 0.5}
            >>> dcat = DCategorical(values)
            >>> dcat.plot()
        """
        plt.bar(range(len(self.values)), list(self.values.values()), tick_label=list(self.values.keys()), **kwargs)
        plt.ylabel("Desirability")
        if non_inform:
            plt.axhline(y=self.missing, linestyle="--", **kwargs)
        plt.show()


class DOverall(DesirabilityBase):
    """
    Combines multiple desirability objects into an overall desirability function.

    This class computes the overall desirability by combining individual desirability objects
    using the geometric mean of their desirability scores.

    Attributes:
        d_objs (list): A list of desirability objects (e.g., DMax, DMin, DTarget, etc.).

    Methods:
        predict(newdata, all=False):
            Predicts the overall desirability based on the individual desirability objects.

    References:
        Many thanks to Max Kuhn for his implementation of the 'desirability' package in R.
        This class is based on the 'desirability' package in R, see:
        https://cran.r-project.org/package=desirability

    Examples:
        >>> from spotpython.utils.desirability import DOverall, DMax, DMin
        >>> import numpy as np
        # Create individual desirability objects
        >>> dmax = DMax(low=0, high=10, scale=1)
        >>> dmin = DMin(low=5, high=15, scale=1)
        # Combine them into an overall desirability object
        >>> doverall = DOverall(dmax, dmin)
        # Predict overall desirability for a set of inputs
        >>> inputs = np.array([[5, 10], [0, 15], [10, 5]])
        >>> overall_desirability = doverall.predict(inputs)
        >>> print(overall_desirability)
        # Predict individual and overall desirabilities
        >>> individual, overall = doverall.predict(inputs, all=True)
        >>> print("Individual:", individual)
        >>> print("Overall:", overall)
    """

    def __init__(self, *d_objs):
        """
        Initializes the DOverall object.

        Args:
            *d_objs (obj): Instances of desirability classes (e.g., DMax, DTarget, etc.).

        Raises:
            ValueError: If any object is not an instance of a valid desirability class.
        """
        valid_classes = (DMax, DMin, DTarget, DArb, DBox, DCategorical)

        if not all(isinstance(obj, valid_classes) for obj in d_objs):
            raise ValueError("All objects must be instances of valid desirability classes.")

        self.d_objs = d_objs  # Store the desirability objects

    def predict(self, newdata, all=False):
        """
        Predicts the overall desirability based on the individual desirability objects.

        Args:
            newdata (list or numpy.ndarray): A list or array of predicted outcomes, one for each desirability object.
            all (bool, optional): Whether to return individual desirabilities along with the overall desirability.
                Defaults to False.

        Returns:
            (float or tuple):
                The overall desirability score, or a tuple of individual and overall desirabilities if `all=True`.

        Raises:
            ValueError: If the number of columns in `newdata` does not match the number of desirability objects.

        Examples:
            >>> from spotpython.utils.desirability import DOverall, DMax, DMin
            >>> import numpy as np
            # Create individual desirability objects
            >>> dmax = DMax(low=0, high=10, scale=1)
            >>> dmin = DMin(low=5, high=15, scale=1)
            >>> doverall = DOverall(dmax, dmin)
            >>> inputs = np.array([[5, 10], [0, 15], [10, 5]])
            >>> overall_desirability = doverall.predict(inputs)
            >>> print(overall_desirability)
        """
        # Ensure newdata is a NumPy array
        newdata = np.array(newdata)

        # Validate the shape of newdata
        if newdata.ndim == 1 and len(newdata) != len(self.d_objs):
            raise ValueError("The number of columns in newdata must match the number of desirability objects.")

        if newdata.ndim == 1:
            newdata = newdata.reshape(1, -1)  # Reshape 1D array to 2D array with one row

        if newdata.shape[1] != len(self.d_objs):
            raise ValueError("The number of columns in newdata must match the number of desirability objects.")

        # Compute individual desirabilities
        individual_desirabilities = [obj.predict(newdata[:, i]) for i, obj in enumerate(self.d_objs)]

        # Compute the geometric mean of the individual desirabilities
        overall_desirability = np.prod(individual_desirabilities, axis=0) ** (1 / len(individual_desirabilities))

        if all:
            return individual_desirabilities, overall_desirability
        return overall_desirability


def conversion_pred(x) -> float:
    """
    Predicts the percent conversion based on the input vector x.

    Args:
        x (list or numpy array): A vector of three input values [x1, x2, x3].

    Returns:
        float: The predicted percent conversion.
    """
    return 81.09 + 1.0284 * x[0] + 4.043 * x[1] + 6.2037 * x[2] - 1.8366 * x[0] ** 2 + 2.9382 * x[1] ** 2 - 5.1915 * x[2] ** 2 + 2.2150 * x[0] * x[1] + 11.375 * x[0] * x[2] - 3.875 * x[1] * x[2]


def activity_pred(x) -> float:
    """
    Predicts the thermal activity based on the input vector x.

    Args:
        x (list or numpy array): A vector of three input values [x1, x2, x3].

    Returns:
        float: The predicted thermal activity.
    """
    return 59.85 + 3.583 * x[0] + 0.2546 * x[1] + 2.2298 * x[2] + 0.83479 * x[0] ** 2 + 0.07484 * x[1] ** 2 + 0.05716 * x[2] ** 2 - 0.3875 * x[0] * x[1] - 0.375 * x[0] * x[2] + 0.3125 * x[1] * x[2]


def rsm_opt(x, d_object, prediction_funcs, space="square") -> float:
    """
    Optimization function to calculate desirability.
    Optimizers minimize, so we return negative desirability.

    Args:
        x (list or np.ndarray): Input parameters (e.g., time, temperature, catalyst).
        d_object (DOverall): Overall desirability object.
        prediction_funcs (list of callables): List of prediction functions to calculate outcomes.
        space (str): Design space ("square" or "circular").

    Returns:
        float: Negative desirability.

    Raises:
        ValueError: If `space` is not "square" or "circular".

    Examples:
        >>> from spotpython.utils.desirability import DOverall, rsm_opt, DTarget
        >>> from spotpython.utils.desirability import conversion_pred, activity_pred
        >>> d_object = DOverall(DTarget(0, 0.5, 1), DTarget(0, 0.5, 1))
        >>> prediction_funcs = [conversion_pred, activity_pred]
        >>> x = [1.0, 2.0, 3.0]
        >>> desirability = rsm_opt(x, d_object, prediction_funcs)
        >>> print(desirability)
        -0.5
    """
    # Calculate predictions for all provided functions
    predictions = [func(x) for func in prediction_funcs]

    # Predict desirability using the overall desirability object
    desirability = d_object.predict(np.array([predictions]))

    # Apply space constraints
    if space == "circular":
        if np.sqrt(np.sum(np.array(x) ** 2)) > 1.682:
            return 0.0
    elif space == "square":
        if np.any(np.abs(np.array(x)) > 1.682):
            return 0.0

    # Return negative desirability
    return -desirability
