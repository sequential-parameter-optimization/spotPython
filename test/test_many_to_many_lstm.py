import pytest
import torch
from spotpython.light.regression.nn_many_to_many_lstm_regressor import ManyToManyLSTM

def test_many_to_many_lstm_initialization():
    # Test initialization with default parameters
    model = ManyToManyLSTM(input_size=10, output_size=1)
    assert isinstance(model, ManyToManyLSTM)
    assert model.lstm_layer.input_size == 10
    assert model.output_layer.out_features == 1

    # Test initialization with custom parameters
    model = ManyToManyLSTM(input_size=5, output_size=2, lstm_units=128, fc_units=64, dropout=0.5, bidirectional=False)
    assert model.lstm_layer.hidden_size == 128
    assert model.fc.out_features == 64
    assert model.output_layer.out_features == 2
    assert model.lstm_layer.bidirectional is False

def test_many_to_many_lstm_forward_pass():
    # Test forward pass with valid input
    model = ManyToManyLSTM(input_size=10, output_size=1)
    x = torch.randn(16, 10, 10)  # Batch of 16 sequences, each of length 10 with 10 features
    lengths = torch.tensor([10] * 16)  # All sequences have length 10
    output = model(x, lengths)
    assert output.shape == (16, 10, 1)

def test_many_to_many_lstm_empty_input():
    # Test forward pass with empty input
    model = ManyToManyLSTM(input_size=10, output_size=1)
    x = torch.empty(0, 10, 10)  # Empty batch
    lengths = torch.tensor([])  # No sequence lengths
    with pytest.raises(ValueError):
        model(x, lengths)

def test_many_to_many_lstm_variable_lengths():
    # Test forward pass with variable sequence lengths
    model = ManyToManyLSTM(input_size=10, output_size=1)
    x = torch.randn(16, 10, 10)  # Batch of 16 sequences
    lengths = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5])  # Variable lengths
    output = model(x, lengths)
    assert output.shape == (16, 10, 1)