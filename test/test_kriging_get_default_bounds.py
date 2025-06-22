import pytest
import numpy as np
from spotpython.surrogate.kriging import Kriging

@pytest.mark.parametrize(
    "method, n_theta, n_p, optim_p, min_theta, max_theta, min_Lambda, max_Lambda, expected_bounds",
    [
        # interpolation, no p optimization
        (
            "interpolation", 2, 1, False, -3.0, 2.0, 1e-9, 1.0,
            [(-3.0, 2.0), (-3.0, 2.0)]
        ),
        # regression, no p optimization
        (
            "regression", 3, 1, False, -2.0, 1.0, 1e-6, 0.5,
            [(-2.0, 1.0), (-2.0, 1.0), (-2.0, 1.0), (np.log10(1e-6), np.log10(0.5))]
        ),
        # reinterpolation, p optimization, n_p=2
        (
            "reinterpolation", 2, 2, True, -3.0, 2.0, 1e-9, 1.0,
            [(-3.0, 2.0), (-3.0, 2.0), (np.log10(1e-9), np.log10(1.0)), (1.0, 2.0), (1.0, 2.0)]
        ),
        # regression, p optimization, n_p=1
        (
            "regression", 1, 1, True, -3.0, 2.0, 1e-9, 1.0,
            [(-3.0, 2.0), (np.log10(1e-9), np.log10(1.0)), (1.0, 2.0)]
        ),
    ]
)
def test_get_default_bounds(
    method, n_theta, n_p, optim_p, min_theta, max_theta, min_Lambda, max_Lambda, expected_bounds
):
    model = Kriging(
        method=method,
        n_theta=n_theta,
        n_p=n_p,
        optim_p=optim_p,
        min_theta=min_theta,
        max_theta=max_theta,
        min_Lambda=min_Lambda,
        max_Lambda=max_Lambda,
    )
    # Set k attribute for fallback in method
    model.k = n_theta
    bounds = model._get_default_bounds()
    # Compare bounds element-wise (floats)
    assert len(bounds) == len(expected_bounds)
    for b, eb in zip(bounds, expected_bounds):
        assert np.allclose(b, eb, rtol=1e-8, atol=1e-12)

def test_get_default_bounds_interpolation_default_params():
    model = Kriging(method="interpolation")
    model.k = 2
    model.n_theta = 2
    bounds = model._get_default_bounds()
    assert bounds == [(-3.0, 2.0), (-3.0, 2.0)]

def test_get_default_bounds_regression_with_lambda():
    model = Kriging(method="regression", n_theta=2, min_Lambda=1e-8, max_Lambda=1e-2)
    model.k = 2
    bounds = model._get_default_bounds()
    expected = [(-3.0, 2.0), (-3.0, 2.0), (np.log10(1e-8), np.log10(1e-2))]
    assert len(bounds) == len(expected)
    for b, eb in zip(bounds, expected):
        assert np.allclose(b, eb)

def test_get_default_bounds_with_optim_p():
    model = Kriging(method="regression", n_theta=2, n_p=2, optim_p=True)
    model.k = 2
    bounds = model._get_default_bounds()
    expected = [
        (-3.0, 2.0), (-3.0, 2.0),
        (np.log10(1e-9), np.log10(1.0)),
        (1.0, 2.0), (1.0, 2.0)
    ]
    assert len(bounds) == len(expected)
    for b, eb in zip(bounds, expected):
        assert np.allclose(b, eb)