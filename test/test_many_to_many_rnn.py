import pytest
import torch
from torch.nn import ReLU
from spotpython.light.regression import ManyToManyRNN, ManyToManyRNNRegressor

def test_ManyToManyRNN_initialization():
    model = ManyToManyRNN(input_size=10, output_size=1, rnn_units=256, fc_units=256, activation_fct=ReLU(), dropout=0.1, bidirectional=True)
    assert model.rnn_layer is not None
    assert model.fc is not None
    assert model.output_layer is not None

def test_ManyToManyRNN_forward():
    model = ManyToManyRNN(input_size=10, output_size=1)
    batch_size = 5
    seq_length = 7
    input_tensor = torch.randn(batch_size, seq_length, 10)
    lengths = torch.tensor([seq_length] * batch_size)
    
    output = model(input_tensor, lengths)
    assert output.shape == (batch_size, seq_length, 1)

def test_ManyToManyRNNRegressor_initialization():
    model = ManyToManyRNNRegressor(_L_in=10, _L_out = 1, batch_size=5)
    assert model.layers is not None
    assert model.example_input_array is not None

@pytest.fixture
def example_batch():
    batch_size = 4
    seq_len = 6
    _L_in = 10
    x = torch.rand(batch_size, seq_len, _L_in)
    lengths = torch.tensor([seq_len] * batch_size)
    y = torch.rand(batch_size, seq_len, 1)
    return (x, lengths, y)

def test_ManyToManyRNNRegressor_forward(example_batch):
    model = ManyToManyRNNRegressor(_L_in=10, _L_out=1, batch_size=4)
    model.eval()  # Set to evaluation mode
    x, lengths, _ = example_batch
    output = model(x, lengths)
    assert output.shape == (4, 6, 1)

def test_ManyToManyRNNRegressor_training_step(example_batch):
    model = ManyToManyRNNRegressor(_L_in=10, _L_out=1, batch_size=4)
    loss = model.training_step(example_batch,0 )
    assert loss is not None

def test_ManyToManyRNNRegressor_validation_step(example_batch):
    model = ManyToManyRNNRegressor(_L_in=10, _L_out=1, batch_size=4)
    x, lengths, y = example_batch
    val_loss = model.validation_step((x, lengths, y), 0)
    assert val_loss is not None

def test_ManyToManyRNNRegressor_configure_optimizers():
    model = ManyToManyRNNRegressor(_L_in=10, _L_out=1, batch_size=4)
    optimizers = model.configure_optimizers()
    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers