import torch


class TorchStandardScaler:
    """
    A class for scaling data using standardization with torch tensors.
    """

    def fit(self, x):
        """
        Compute the mean and standard deviation of the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Raises:
            TypeError: If the input is not a torch tensor.
        """
        if not torch.is_tensor(x):
            raise TypeError("Input should be a torch tensor")
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        """
        Scale the input tensor using the computed mean and standard deviation.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The scaled tensor.
        Raises:
            TypeError: If the input is not a torch tensor.
            RuntimeError: If the scaler has not been fitted before transforming data.
        """
        if not torch.is_tensor(x):
            raise TypeError("Input should be a torch tensor")
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            raise RuntimeError("Must fit scaler before transforming data")
        x = (x - self.mean) / (self.std + 1e-7)
        return x

    def fit_transform(self, x):
        """
        Fit the scaler to the input tensor and then scale the tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The scaled tensor.
        """
        self.fit(x)
        return self.transform(x)


class TorchMinMaxScaler:
    """
    A class for scaling data using min-max normalization with PyTorch tensors.
    """

    def fit(self, x):
        """
        Fit the scaler to the input data.
        Parameters:
        - x: torch.Tensor
            The input data to fit the scaler to.
        Raises:
        - TypeError: If the input is not a torch tensor.
        """
        if not torch.is_tensor(x):
            raise TypeError("Input should be a torch tensor")
        self.min = x.min(0, keepdim=True).values
        self.max = x.max(0, keepdim=True).values

    def transform(self, x):
        """
        Transform the input data using the fitted scaler.
        Parameters:
        - x: torch.Tensor
            The input data to transform.
        Returns:
        - torch.Tensor: The transformed data.
        Raises:
        - TypeError: If the input is not a torch tensor.
        - RuntimeError: If the scaler has not been fitted before transforming data.
        """
        if not torch.is_tensor(x):
            raise TypeError("Input should be a torch tensor")
        if not hasattr(self, "min") or not hasattr(self, "max"):
            raise RuntimeError("Must fit scaler before transforming data")
        x = (x - self.min) / (self.max - self.min + 1e-7)
        return x

    def fit_transform(self, x):
        """
        Fit the scaler to the input data and transform it.
        Parameters:
        - x: torch.Tensor
            The input data to fit and transform.
        Returns:
        - torch.Tensor: The transformed data.
        Raises:
        - TypeError: If the input is not a torch tensor.
        """
        self.fit(x)
        return self.transform(x)
