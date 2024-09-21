import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import lightning as L
from spotpython.light.regression import NNLinearRegressor

# Explanation of the Tests:
#
# 1. Initialization Test:
#    - This verifies that the `NNLinearRegressor` class is initialized with the correct parameters and
#       that the layers attribute is created properly.
# 2. Forward Pass Test:
#    - This checks that the forward pass of the model produces an output of the expected shape given
#       a batch of input data.
# 3. Training Step Test:
#    - This simulates a training epoch and ensures that each training step produces a valid loss value.
# 4. Validation Step Test:
#    - This checks the validation step during training. It validates that each validation step produces
#       a valid loss value.
# 5. Testing Step Test:
#    - This checks the test step to ensure that the model can correctly evaluate the loss on the
#   test dataset.
# 6. Prediction Step Test:
#    - This tests the prediction step. It checks that the prediction step returns the correct tensors.
# 7. Optimizer Configuration Test:
#    - This ensures the optimizer is correctly configured with the provided parameters.


# Sample data for testing
n_samples, n_features = 100, 10
x = torch.randn(n_samples, n_features)
y = torch.randn(n_samples)
dataset = TensorDataset(x, y)
data_loader = DataLoader(dataset, batch_size=8)

# Parameters for NNLinearRegressor
params = {
    "l1": 128,
    "epochs": 10,
    "batch_size": 8,
    "initialization": "xavier",
    "act_fn": nn.ReLU(),
    "optimizer": "Adam",
    "dropout_prob": 0.1,
    "lr_mult": 0.1,
    "patience": 5,
    "_L_in": 10,
    "_L_out": 1,
    "_torchmetric": "mean_squared_error",
}


@pytest.fixture
def nn_linear_regressor():
    return NNLinearRegressor(**params)


def test_initialization(nn_linear_regressor):
    # Verify initialization
    assert nn_linear_regressor.hparams.l1 == params["l1"]
    assert isinstance(nn_linear_regressor.layers, nn.Sequential)


def test_forward_pass(nn_linear_regressor):
    batch_x, _ = next(iter(data_loader))
    output = nn_linear_regressor(batch_x)
    assert output.shape == (batch_x.shape[0], params["_L_out"])


def test_training_step(nn_linear_regressor):
    trainer = L.Trainer(max_epochs=1, enable_checkpointing=False, accelerator="cpu")
    train_loader = data_loader
    trainer.fit(nn_linear_regressor, train_loader)
    batch_x, batch_y = next(iter(train_loader))
    losses = []
    for x, y in train_loader:
        loss = nn_linear_regressor.training_step((x, y))
        losses.append(loss.item())

    assert len(losses) == len(train_loader)
    assert all(isinstance(loss, float) for loss in losses)


def test_validation_step(nn_linear_regressor):
    trainer = L.Trainer(max_epochs=1, enable_checkpointing=False, accelerator="cpu")
    val_loader = data_loader
    trainer.validate(nn_linear_regressor, val_loader)

    # validation_step is tested within the context of trainer.validate
    val_losses = []
    for x, y in val_loader:
        val_loss = nn_linear_regressor._calculate_loss((x, y))
        val_losses.append(val_loss.item())

    assert len(val_losses) == len(val_loader)
    assert all(isinstance(loss, float) for loss in val_losses)


def test_testing_step(nn_linear_regressor):
    trainer = L.Trainer(max_epochs=1, enable_checkpointing=False, accelerator="cpu")
    test_loader = data_loader
    trainer.test(nn_linear_regressor, test_loader)

    # test_step is tested within the context of trainer.test
    test_losses = []
    for x, y in test_loader:
        test_loss = nn_linear_regressor._calculate_loss((x, y))
        test_losses.append(test_loss.item())

    assert len(test_losses) == len(test_loader)
    assert all(isinstance(loss, float) for loss in test_losses)


def test_predict_step(nn_linear_regressor):
    batch_x, batch_y = next(iter(data_loader))
    output = nn_linear_regressor.predict_step((batch_x, batch_y), 0)
    assert len(output) == 3  # x, y, yhat
    assert isinstance(output[0], torch.Tensor)
    assert isinstance(output[1], torch.Tensor)
    assert isinstance(output[2], torch.Tensor)


def test_configure_optimizers(nn_linear_regressor):
    res  = nn_linear_regressor.configure_optimizers()
    # res is a dictionary containing the optimizer and the scheduler
    optimizer = res["optimizer"]
    assert isinstance(optimizer, torch.optim.Optimizer)
