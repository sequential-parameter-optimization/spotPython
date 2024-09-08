import pytest
import torch
from spotpython.utils.scaler import TorchStandardScaler


def test_fit():
    """Test the `fit` method for correct mean and std computation."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_mean = torch.tensor([[2.0, 3.0]])
    expected_std = torch.tensor([[1.0, 1.0]])

    scaler = TorchStandardScaler()
    scaler.fit(tensor)

    torch.testing.assert_close(scaler.mean, expected_mean)
    torch.testing.assert_close(scaler.std, expected_std, atol=1e-7, rtol=1e-7)


def test_transform():
    """Test the `transform` method for correct data scaling."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaler = TorchStandardScaler()
    scaler.fit(tensor)
    transformed = scaler.transform(tensor)

    expected_transformed = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])

    torch.testing.assert_close(transformed, expected_transformed)


def test_fit_transform():
    """Test the `fit_transform` method for combined fitting and transforming."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaler = TorchStandardScaler()
    transformed = scaler.fit_transform(tensor)

    expected_transformed = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])

    torch.testing.assert_close(transformed, expected_transformed)


def test_input_not_tensor():
    """Test that a TypeError is raised if the input data is not a tensor."""
    scaler = TorchStandardScaler()
    with pytest.raises(TypeError):
        scaler.fit([1.0, 2.0])  # Passing a list instead of a tensor


def test_unfitted_transform():
    """Test that a RuntimeError is raised if attempting to transform without fitting first."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaler = TorchStandardScaler()

    with pytest.raises(RuntimeError):
        scaler.transform(tensor)
