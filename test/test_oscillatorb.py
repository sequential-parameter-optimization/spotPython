import pytest
import torch
import numpy as np
from spotpython.pinns.solvers import oscillatorb

def test_oscillatorb_basic_properties():
    """
    Tests basic properties of the oscillatorb output:
    - Output tensor shapes
    - Initial time and y value
    - Final time value
    - Data types of output tensors
    """
    n_steps = 100
    t_min = 0.0
    t_max = 10.0
    y0 = 1.0
    alpha = 0.1
    omega = np.pi / 2

    t_vals, y_vals = oscillatorb(
        n_steps=n_steps,
        t_min=t_min,
        t_max=t_max,
        y0=y0,
        alpha=alpha,
        omega=omega
    )

    # Check shapes
    assert t_vals.shape == (n_steps, 1), f"Expected t_vals shape ({n_steps}, 1), got {t_vals.shape}"
    assert y_vals.shape == (n_steps, 1), f"Expected y_vals shape ({n_steps}, 1), got {y_vals.shape}"

    # Check data types
    assert t_vals.dtype == torch.float32, f"Expected t_vals dtype torch.float32, got {t_vals.dtype}"
    assert y_vals.dtype == torch.float32, f"Expected y_vals dtype torch.float32, got {y_vals.dtype}"

    # Check initial conditions
    assert t_vals[0].item() == pytest.approx(t_min), f"Expected initial t to be {t_min}, got {t_vals[0].item()}"
    assert y_vals[0].item() == pytest.approx(y0), f"Expected initial y to be {y0}, got {y_vals[0].item()}"

    # Check final time point
    # t_step = (t_max - t_min) / n_steps
    # expected_t_final = t_min + (n_steps - 1) * t_step
    # Using the formula from the source directly for t_points generation:
    # t_points = np.arange(t_min, t_min + n_steps * t_step, t_step)[:n_steps]
    # The last element of this arange will be t_min + (n_steps - 1) * t_step
    t_step_calc = (t_max - t_min) / n_steps
    expected_t_final_calc = t_min + (n_steps - 1) * t_step_calc
    assert t_vals[-1].item() == pytest.approx(expected_t_final_calc), \
        f"Expected final t to be {expected_t_final_calc}, got {t_vals[-1].item()}"

def test_oscillatorb_no_damping_no_forcing():
    """
    Tests the case where alpha = 0 and omega = 0.
    In this scenario, y' = 0, so y(t) should remain y0.
    """
    n_steps = 50
    t_min = 0.0
    t_max = 5.0
    y0 = 2.5
    alpha = 0.0
    omega = 0.0  # This makes sin(omega*t) = 0

    t_vals, y_vals = oscillatorb(
        n_steps=n_steps,
        t_min=t_min,
        t_max=t_max,
        y0=y0,
        alpha=alpha,
        omega=omega
    )

    # y should remain constant at y0
    expected_y_values = torch.full((n_steps, 1), y0, dtype=torch.float32)
    assert torch.allclose(y_vals, expected_y_values, atol=1e-6), \
        f"Expected all y values to be {y0}, got {y_vals.numpy().flatten()}"

def test_oscillatorb_single_step():
    """
    Tests the behavior when n_steps = 1.
    The output should contain only the initial condition.
    """
    n_steps = 1
    t_min = 1.0
    t_max = 10.0  # t_max doesn't really matter here beyond t_step calculation
    y0 = -1.5
    alpha = 0.1
    omega = np.pi

    t_vals, y_vals = oscillatorb(
        n_steps=n_steps,
        t_min=t_min,
        t_max=t_max,
        y0=y0,
        alpha=alpha,
        omega=omega
    )

    # Check shapes
    assert t_vals.shape == (1, 1), f"Expected t_vals shape (1, 1) for n_steps=1, got {t_vals.shape}"
    assert y_vals.shape == (1, 1), f"Expected y_vals shape (1, 1) for n_steps=1, got {y_vals.shape}"

    # Check values (should be initial conditions)
    assert t_vals[0].item() == pytest.approx(t_min), \
        f"Expected t to be {t_min} for n_steps=1, got {t_vals[0].item()}"
    assert y_vals[0].item() == pytest.approx(y0), \
        f"Expected y to be {y0} for n_steps=1, got {y_vals[0].item()}"

@pytest.mark.parametrize("n_steps_val", [2, 10, 50])
@pytest.mark.parametrize("y0_val", [0.0, 10.0, -5.0])
def test_oscillatorb_various_inputs(n_steps_val, y0_val):
    """
    Tests with a few different n_steps and y0 values to ensure basic execution.
    """
    t_min = 0.0
    t_max = 1.0
    alpha = 0.05
    omega = np.pi / 4.0

    t_vals, y_vals = oscillatorb(
        n_steps=n_steps_val,
        t_min=t_min,
        t_max=t_max,
        y0=y0_val,
        alpha=alpha,
        omega=omega
    )

    assert t_vals.shape == (n_steps_val, 1)
    assert y_vals.shape == (n_steps_val, 1)
    assert t_vals[0].item() == pytest.approx(t_min)
    assert y_vals[0].item() == pytest.approx(y0_val)
    t_step_calc = (t_max - t_min) / n_steps_val
    if n_steps_val > 0 : # Avoid division by zero if n_steps_val could be 0
      expected_t_final_calc = t_min + (n_steps_val - 1) * t_step_calc
      if n_steps_val == 1: # np.arange behavior for single step
          expected_t_final_calc = t_min
      assert t_vals[-1].item() == pytest.approx(expected_t_final_calc)
    elif n_steps_val == 0: # Should ideally raise error or handle, current code might produce empty
        assert t_vals.numel() == 0 # Expect empty tensor
        assert y_vals.numel() == 0 # Expect empty tensor
