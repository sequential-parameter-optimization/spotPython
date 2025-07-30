import numpy as np
from spotpython.utils.metrics import calculate_xai_consistency


def test_xai_consistency():
    # Mock data for testing

    dl_attr_test_sum = [1, 2, 3, 4, 5]
    row_sum_dl = np.sum(dl_attr_test_sum, axis=0)
    if row_sum_dl == 0:
        row_sum_dl += 1e-10
    scaled_attribution_dl = dl_attr_test_sum / row_sum_dl

    ig_attr_test_sum = [1, 2, 3, 4, 5]
    row_sum_ig = np.sum(ig_attr_test_sum, axis=0)
    if row_sum_ig == 0:
        row_sum_ig += 1e-10
    scaled_attribution_ig = ig_attr_test_sum / row_sum_ig

    attributions = [scaled_attribution_dl, scaled_attribution_ig]
    result = calculate_xai_consistency(attributions)
    print("XAI Consistency Result:")
    print(result)

    # Assert that the result is 1
    assert abs(result - 1) < 1e-10


def test_xai_consistency_negative():
    # Mock data for testing negative consistency

    dl_attr_test_sum = [1, 2, 3, 4, 5]
    row_sum_dl = np.sum(dl_attr_test_sum, axis=0)
    if row_sum_dl == 0:
        row_sum_dl += 1e-10
    scaled_attribution_dl = dl_attr_test_sum / row_sum_dl

    ig_attr_test_sum = [-1, -2, -3, -4, -5]
    row_sum_ig = np.sum(np.abs(ig_attr_test_sum), axis=0)
    if row_sum_ig == 0:
        row_sum_ig += 1e-10
    scaled_attribution_ig = ig_attr_test_sum / row_sum_ig

    attributions = [scaled_attribution_dl, scaled_attribution_ig]
    result = calculate_xai_consistency(attributions)
    print("XAI Consistency Result (Negative):")
    print(result)

    # Assert that the result is -1
    assert abs(result + 1) < 1e-10