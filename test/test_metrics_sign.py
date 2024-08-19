from spotpython.utils.metrics import get_metric_sign
import pytest


def test_get_metric_sign():
    # Test for accuracy_score
    assert get_metric_sign("accuracy_score") == -1

    # Test for hamming_loss
    assert get_metric_sign("hamming_loss") == +1

    # Test for f1_score
    assert get_metric_sign("f1_score") == -1

    # Test for roc_auc_score
    assert get_metric_sign("roc_auc_score") == -1

    # Test for unknown metric
    with pytest.raises(ValueError):
        get_metric_sign("unknown_metric")
