from spotpython.utils.init import fun_control_init


def test_fun_control_init():
    """Test fun_control_init function."""
    fun_control = fun_control_init(_L_in=64, _L_out=11, num_workers=0, device=None)
    assert fun_control["_L_in"] == 64
    assert fun_control["_L_out"] == 11
    assert fun_control["num_workers"] == 0
    assert fun_control["device"] is None
    assert fun_control["task"] is None
    assert fun_control["sigma"] == 0.0
    assert fun_control["CHECKPOINT_PATH"] == "runs/saved_models/"
    assert fun_control["DATASET_PATH"] == "data/"
    assert fun_control["RESULTS_PATH"] == "results/"
    assert fun_control["TENSORBOARD_PATH"] == "runs/"
    assert fun_control["spot_tensorboard_path"] is None
