import numpy as np
import torch.nn as nn
import pytest
from spotpython.torch.dimensions import extract_linear_dims

class NNLinearRegressor(nn.Module):
    def __init__(self):
        super(NNLinearRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(2, 1),
        )

def test_extract_linear_dims():
    net = NNLinearRegressor()
    expected_dims = np.array([10, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 1])
    result = extract_linear_dims(net)
    assert np.array_equal(result, expected_dims), f"Expected {expected_dims}, but got {result}"

if __name__ == "__main__":
    pytest.main([__file__])