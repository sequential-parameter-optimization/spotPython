import pytest
import torch
from torch.utils.data import TensorDataset
from unittest.mock import MagicMock

# Assuming the class containing transform_dataset is named MyDataModule
class MyDataModule:
    def __init__(self, scaler):
        self.scaler = scaler

    def transform_dataset(self, dataset):
        """Applies the scaler transformation to the dataset.

        Args:
            dataset (List[Tuple[torch.Tensor, Any]]): The dataset to transform, consisting of data and target pairs.

        Returns:
            TensorDataset: A PyTorch TensorDataset containing the transformed and cloned data and targets.

        Raises:
            ValueError: If the input data is not correctly formatted for transformation.
        """
        try:
            # Perform transformations on the data in a single iteration
            transformed_data = [(self.scaler.transform(data), target) for data, target in dataset]

            # Clone and detach data tensors
            data_tensors = [data.clone().detach() for data, _ in transformed_data]
            target_tensors = [target.clone().detach() for _, target in transformed_data]

            # Create a TensorDataset from the processed data
            return TensorDataset(torch.stack(data_tensors).squeeze(1), torch.stack(target_tensors))

        except Exception as e:
            raise ValueError(f"Error transforming dataset: {e}")


# Test function for transform_dataset
@pytest.fixture
def setup_data():
    # Mock dataset
    input_data = torch.randn(3, 4)  # Mock input data
    target_data = torch.tensor([0, 1, 2])  # Mock target data

    dataset = [(input_data[i], target_data[i]) for i in range(len(target_data))]
    
    # Mock scaler with a simple transform logic
    mock_scaler = MagicMock()
    mock_scaler.transform = lambda x: 2 * x  # Example transformation: multiply by 2

    return mock_scaler, dataset

def test_transform_dataset(setup_data):
    mock_scaler, dataset = setup_data
    data_module = MyDataModule(mock_scaler)
    
    transformed_dataset = data_module.transform_dataset(dataset)

    # Check that transform_dataset returns a TensorDataset
    assert isinstance(transformed_dataset, TensorDataset)

    # Extract transformed data and targets
    transformed_data, transformed_targets = transformed_dataset.tensors

    # Verify the shape
    assert transformed_data.shape == torch.Size([3, 4])
    assert transformed_targets.shape == torch.Size([3])

    # Verify that the data was transformed correctly (i.e., multiplied by 2)
    expected_data = torch.stack([mock_scaler.transform(d[0]) for d in dataset]).squeeze(1)
    for td, ed in zip(transformed_data, expected_data):
        assert torch.allclose(td, ed)

    # Verify that the targets were unchanged
    expected_targets = torch.tensor([d[1] for d in dataset])
    assert torch.equal(transformed_targets, expected_targets)