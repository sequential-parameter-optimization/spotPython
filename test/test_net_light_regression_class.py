import pytest
from torch.utils.data import DataLoader
from spotPython.data.diabetes import Diabetes
from spotPython.light.regression.netlightregression import NetLightRegression
from torch import nn
import lightning as L


def test_net_light_regression_class():
    BATCH_SIZE = 8

    dataset = Diabetes()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    net_light_regression = NetLightRegression(l1=128,
                                        epochs=10,
                                        batch_size=BATCH_SIZE,
                                        initialization='xavier',
                                        act_fn=nn.ReLU(),                                   optimizer='Adam',
                                        dropout_prob=0.1,
                                        lr_mult=0.1,                                  patience=5, 
                                        _L_in=10,
                                        _L_out=1)
    trainer = L.Trainer(max_epochs=2,  enable_progress_bar=False)
    trainer.fit(net_light_regression, train_loader, val_loader)    
    res = trainer.test(net_light_regression, test_loader)
    # test if the entry 'hp_metric' is in the res dict
    assert 'hp_metric' in res[0].keys()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
