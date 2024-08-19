from torch.utils.data import DataLoader
from spotpython.data.diabetes import Diabetes
from spotpython.light.regression.netlightregression import NetLightRegression
from torch import nn
import lightning as L


def test_optimizer_handler_adam():
    BATCH_SIZE = 8
    lr_mult = 0.1

    dataset = Diabetes()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    net_light_base = NetLightRegression(
        l1=128,
        epochs=10,
        batch_size=BATCH_SIZE,
        initialization="xavier",
        act_fn=nn.ReLU(),
        optimizer="Adam",
        dropout_prob=0.1,
        lr_mult=lr_mult,
        patience=5,
        _L_in=10,
        _L_out=1,
        _torchmetric="mean_squared_error",
    )
    trainer = L.Trainer(accelerator="cpu", max_epochs=2, enable_progress_bar=False)
    trainer.fit(net_light_base, train_loader, val_loader)
    # Adam uses a lr which is calculated as lr=lr_mult * 0.001, so this value
    # should be 0.1 * 0.001 = 0.0001
    assert trainer.optimizers[0].param_groups[0]["lr"] == lr_mult * 0.001


def test_optimizer_handler_adadelta():
    BATCH_SIZE = 8
    lr_mult = 0.1
    dataset = Diabetes()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    net_light_base = NetLightRegression(
        l1=128,
        epochs=10,
        batch_size=BATCH_SIZE,
        initialization="xavier",
        act_fn=nn.ReLU(),
        optimizer="Adadelta",
        dropout_prob=0.1,
        lr_mult=lr_mult,
        patience=5,
        _L_in=10,
        _L_out=1,
        _torchmetric="mean_squared_error",
    )
    trainer = L.Trainer(accelerator="cpu", max_epochs=2, enable_progress_bar=False)
    trainer.fit(net_light_base, train_loader, val_loader)
    # Adadelta uses a lr which is calculated as lr=lr_mult * 1.0, so this value
    # should be 1.0 * 0.1 = 0.1
    assert trainer.optimizers[0].param_groups[0]["lr"] == lr_mult * 1.0
