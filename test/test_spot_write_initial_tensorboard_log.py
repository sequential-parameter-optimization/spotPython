import numpy as np
from unittest.mock import patch
from spotpython.spot.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init

def simple_fun(X, **kwargs):
    return np.sum(X, axis=1)

@patch("spotpython.spot.spot.SummaryWriter")
def test_write_initial_tensorboard_log_calls_writer(mock_writer):
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        tensorboard_log=True
    )
    fun_control["spot_tensorboard_path"] = "runs/test_spot"
    design_control = design_control_init(init_size=3)
    spot = Spot(fun=simple_fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    spot.evaluate_initial_design()  # <-- This line is required!
    spot.spot_writer = mock_writer.return_value
    spot.write_initial_tensorboard_log()
    assert spot.spot_writer.add_hparams.call_count == spot.X.shape[0]
    assert spot.spot_writer.flush.call_count == spot.X.shape[0]

def test_write_initial_tensorboard_log_no_writer():
    fun_control = fun_control_init(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        tensorboard_log=False
    )
    fun_control["spot_tensorboard_path"] = None
    design_control = design_control_init(init_size=2)
    spot = Spot(fun=simple_fun, fun_control=fun_control, design_control=design_control)
    spot.initialize_design_matrix()
    spot.spot_writer = None
    spot.write_initial_tensorboard_log()