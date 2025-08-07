import numpy as np
from spotpython.utils.metrics import calculate_xai_consistency_corr, calculate_xai_consistency_cosine, calculate_xai_consistency_euclidean


def test_xai_consistency_corr():
    # Mock data for testing

    dl_attr_test_sum = [1, 2, 3, 4, 5]
    l2_norm = np.linalg.norm(dl_attr_test_sum)
    scaled_attribution_dl = dl_attr_test_sum / l2_norm if l2_norm != 0 else dl_attr_test_sum

    ig_attr_test_sum = [1, 2, 3, 4, 5]
    l2_norm = np.linalg.norm(ig_attr_test_sum)
    scaled_attribution_ig = ig_attr_test_sum / l2_norm if l2_norm != 0 else ig_attr_test_sum

    attributions = [scaled_attribution_dl, scaled_attribution_ig]
    result = calculate_xai_consistency_corr(attributions)
    print("XAI Consistency Result:")
    print(result)

    # Assert that the result is 1
    assert abs(result - 1) < 1e-10


def test_xai_consistency_negative_corr():
    # Mock data for testing negative consistency

    dl_attr_test_sum = [1, 2, 3, 4, 5]
    l2_norm = np.linalg.norm(dl_attr_test_sum)
    scaled_attribution_dl = dl_attr_test_sum / l2_norm if l2_norm != 0 else dl_attr_test_sum

    ig_attr_test_sum = [-2, -3, -4, -5, -6]
    l2_norm = np.linalg.norm(ig_attr_test_sum)
    scaled_attribution_ig = ig_attr_test_sum / l2_norm if l2_norm != 0 else ig_attr_test_sum

    attributions = [scaled_attribution_dl, scaled_attribution_ig]
    result = calculate_xai_consistency_corr(attributions)
    print("XAI Consistency Result (Negative):")
    print(result)

    # Assert that the result is -1
    assert abs(result + 1) < 1e-10


def test_xai_consistency_cosine():
    # Mock data for testing cosine consistency

    dl_attr_test_sum = [1, 2, 3, 4, 5]
    l2_norm = np.linalg.norm(dl_attr_test_sum)
    scaled_attribution_dl = dl_attr_test_sum / l2_norm if l2_norm != 0 else dl_attr_test_sum
    ig_attr_test_sum = [1, 2, 3, 4, 5]
    l2_norm = np.linalg.norm(ig_attr_test_sum)
    scaled_attribution_ig = ig_attr_test_sum / l2_norm if l2_norm != 0 else ig_attr_test_sum

    attributions = [scaled_attribution_dl, scaled_attribution_ig]
    result = calculate_xai_consistency_cosine(attributions)
    print("XAI Consistency Cosine Result:")
    print(result)
    # Assert that the result is 1
    assert abs(result - 1) < 1e-10


def test_xai_consistency_negative_cosine():
    # Mock data for testing negative cosine consistency

    dl_attr_test_sum = [1, 2, 3, 4, 5]
    l2_norm = np.linalg.norm(dl_attr_test_sum)
    scaled_attribution_dl = dl_attr_test_sum / l2_norm if l2_norm != 0 else dl_attr_test_sum
    ig_attr_test_sum = [-1, -2, -3, -4, -5]
    l2_norm = np.linalg.norm(ig_attr_test_sum)
    scaled_attribution_ig = ig_attr_test_sum / l2_norm if l2_norm != 0 else ig_attr_test_sum

    attributions = [scaled_attribution_dl, scaled_attribution_ig]
    result = calculate_xai_consistency_cosine(attributions)
    print("XAI Consistency Cosine Result (Negative):")
    print(result)

    # Assert that the result is -1
    assert abs(result + 1) < 1e-10


def test_xai_consistency_euclidean():
    # Mock data for testing Euclidean consistency

    dl_attr_test_sum = [1, 2, 3, 4, 5]
    l2_norm = np.linalg.norm(dl_attr_test_sum)
    scaled_attribution_dl = dl_attr_test_sum / l2_norm if l2_norm != 0 else dl_attr_test_sum
    ig_attr_test_sum = [1, 2, 3, 4, 5]
    l2_norm = np.linalg.norm(ig_attr_test_sum)
    scaled_attribution_ig = ig_attr_test_sum / l2_norm if l2_norm != 0 else ig_attr_test_sum

    attributions = [scaled_attribution_dl, scaled_attribution_ig]
    result = calculate_xai_consistency_euclidean(attributions)
    print("XAI Consistency Euclidean Result:")
    print(result)

    # Assert that the result is close to zero
    assert abs(result) < 1e-10


def test_xai_consistency_negative_euclidean():
    # Mock data for testing negative Euclidean consistency

    dl_attr_test_sum = [1, 2, 3, 4, 5]
    l2_norm = np.linalg.norm(dl_attr_test_sum)
    scaled_attribution_dl = dl_attr_test_sum / l2_norm if l2_norm != 0 else dl_attr_test_sum
    ig_attr_test_sum = [-1, -2, -3, -4, -5]
    l2_norm = np.linalg.norm(ig_attr_test_sum)
    scaled_attribution_ig = ig_attr_test_sum / l2_norm if l2_norm != 0 else ig_attr_test_sum

    attributions = [scaled_attribution_dl, scaled_attribution_ig]
    result = calculate_xai_consistency_euclidean(attributions)
    print("XAI Consistency Euclidean Result (Negative):")
    print(result)

    # Assert that the result is close to two
    assert abs(result - 2) < 1e-10
