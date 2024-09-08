import pytest
import torch
from spotpython.utils.scaler import TorchMinMaxScaler


def test_min_max_scaler_fit():
    """Test the min and max values computed by the `fit` method."""
    tensor = torch.tensor([[2.0, 4.0], [1.0, 5.0], [3.0, 6.0]])
    expected_min = torch.tensor([[1.0, 4.0]])
    expected_max = torch.tensor([[3.0, 6.0]])

    scaler = TorchMinMaxScaler()
    scaler.fit(tensor)

    torch.testing.assert_close(scaler.min, expected_min)
    torch.testing.assert_close(scaler.max, expected_max)


def test_min_max_scaler_transform():
    """Test the output of the `transform` method."""
    tensor = torch.tensor([[2.0, 4.0], [1.0, 5.0], [3.0, 6.0]])
    scaler = TorchMinMaxScaler()
    scaler.fit(tensor)
    transformed = scaler.transform(tensor)

    expected_transformed = torch.tensor([[0.5, 0.0], [0.0, 0.5], [1.0, 1.0]])

    torch.testing.assert_close(transformed, expected_transformed)


def test_min_max_scaler_fit_transform():
    """Check that `fit_transform` method correctly fits and transforms the data."""
    tensor = torch.tensor([[2.0, 4.0], [1.0, 5.0], [3.0, 6.0]])
    scaler = TorchMinMaxScaler()
    transformed = scaler.fit_transform(tensor)

    expected_transformed = torch.tensor([[0.5, 0.0], [0.0, 0.5], [1.0, 1.0]])

    torch.testing.assert_close(transformed, expected_transformed)


def test_input_validation():
    """Ensure type error is raised with incorrect input type."""
    scaler = TorchMinMaxScaler()
    with pytest.raises(TypeError):
        scaler.fit([[1, 2], [3, 4]])  # Not a tensor, should raise error


def test_transform_before_fit():
    """Ensure appropriate error is raised when transform is called before fit."""
    scaler = TorchMinMaxScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(torch.tensor([[2.0, 4.0], [1.0, 5.0]]))
