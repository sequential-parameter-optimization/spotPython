import pytest
import torch
from spotpython.light.regression.nn_many_to_many_gru_regressor import ManyToManyGRU


def test_many_to_many_gru_initialization():
    """Test initialization of the ManyToManyGRU model."""
    model = ManyToManyGRU(input_size=10, output_size=1, gru_units=128, fc_units=64, dropout=0.1, bidirectional=False)
    assert isinstance(model, ManyToManyGRU)
    assert model.gru_layer.input_size == 10
    assert model.gru_layer.hidden_size == 128
    assert model.fc.in_features == 128
    assert model.fc.out_features == 64
    assert model.output_layer.out_features == 1


def test_many_to_many_gru_forward_pass():
    """Test forward pass with valid input."""
    model = ManyToManyGRU(input_size=10, output_size=1)
    x = torch.randn(16, 10, 10)  # Batch of 16 sequences, each of length 10 with 10 features
    lengths = torch.tensor([10] * 16)  # All sequences have length 10
    output = model(x, lengths)
    assert output.shape == (16, 10, 1)  # Output shape should match (batch_size, seq_len, output_size)


def test_many_to_many_gru_empty_input():
    """Test forward pass with empty input."""
    model = ManyToManyGRU(input_size=10, output_size=1)
    x = torch.empty(0, 10, 10)  # Empty batch
    lengths = torch.tensor([])  # No sequence lengths
    with pytest.raises(ValueError):
        model(x, lengths)


def test_many_to_many_gru_variable_lengths():
    """Test forward pass with variable sequence lengths."""
    model = ManyToManyGRU(input_size=10, output_size=1)
    x = torch.randn(16, 10, 10)  # Batch of 16 sequences
    lengths = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5])  # Variable lengths
    output = model(x, lengths)
    assert output.shape == (16, 10, 1)  # Output shape should match (batch_size, seq_len, output_size)