import numpy as np
import torch
from typing import Tuple


def oscillatorb(n_steps: int = 3000, t_min: float = 0.0, t_max: float = 30.0, y0: float = 1.0, alpha: float = 0.1, omega: float = np.pi / 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solves the first-order ODE y' = -alpha*y + sin(omega*t) using a
    two-stage explicit Runge-Kutta method as described in the reference.
    The ODE represents a damped harmonic oscillator with a sine forcing term.

    The specific numerical scheme used is:
    1. y_intermediate = y_current + (t_step/2) * f(t_current + t_step/2, y_current)
    2. y_next = y_current + t_step * f(t_current + t_step, y_intermediate)
    where f(t,y) = -alpha*y + sin(omega*t). This is a second-order method.

    Args:
        n_steps (int): Number of time points in the discretized time domain,
            including the initial point. Defaults to 3000.
        t_min (float): Initial time. Defaults to 0.0.
        t_max (float): Defines the nominal end of the time interval. The time step
            is calculated as (t_max - t_min) / n_steps. The actual last
            time point will be t_min + (n_steps - 1) * t_step. Defaults to 30.0.
        y0 (float): Initial condition for y at t_min. Defaults to 1.0.
        alpha (float): Damping coefficient in the ODE. Defaults to 0.1.
        omega (float): Angular frequency for the sine forcing term in the ODE.
            Defaults to np.pi / 2.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two PyTorch tensors:
            - t_points_tensor: Tensor of time points, shape (n_steps, 1).
            - y_values_tensor: Tensor of corresponding y values, shape (n_steps, 1).

    Examples:
        >>> from spotpython.pinns.solvers import oscillatorb
        >>> import numpy as np
        >>> import torch
        >>> t_vals, y_vals = oscillatorb(n_steps=100, t_min=0.0, t_max=10.0, y0=1.0, alpha=0.1, omega=np.pi/2)
        >>> print(t_vals.shape, y_vals.shape)
        torch.Size([100, 1]) torch.Size([100, 1])
        >>> print(f"Initial t: {t_vals[0].item():.2f}, Initial y: {y_vals[0].item():.2f}")
        Initial t: 0.00, Initial y: 1.00
        >>> # Last t will be t_max - t_step for this configuration
        >>> # t_step = (10.0 - 0.0) / 100 = 0.1
        >>> # Last t = 0.0 + (100-1)*0.1 = 9.9
        >>> print(f"Final t: {t_vals[-1].item():.2f}, Final y: {y_vals[-1].item():.2f}")
        Final t: 9.90, Final y: ...

    References:
        - Solving differential equations using physics informed deep learning: a hand-on tutorial with benchmark tests. Baty, Hubert and Baty, Leo. April 2023.
    """
    t_step = (t_max - t_min) / n_steps  # Time step
    # Time points: t_min, t_min + t_step, ..., t_min + (n_steps-1)*t_step
    t_points = np.arange(t_min, t_min + n_steps * t_step, t_step)[:n_steps]

    y = [y0]  # List to store y values, starting with initial condition

    # Solve for the time evolution
    # t_points[0] corresponds to y0. Loop starts from t_points[1].
    for t_current_step_end in t_points[1:]:
        # t_midpoint is the midpoint of the current integration interval
        # Interval: [t_current_step_end - t_step, t_current_step_end]
        # Midpoint: (t_current_step_end - t_step) + t_step/2 = t_current_step_end - t_step/2
        t_midpoint = t_current_step_end - t_step / 2.0
        # y_prev is the last computed value of y
        y_prev = y[-1]

        # Stage 1: Calculate intermediate y value (y_intermediate)
        # Uses slope at t_midpoint, with y_prev
        # f(t,y) = -alpha*y + sin(omega*t)
        slope_at_t_mid_using_y_prev = -alpha * y_prev + np.sin(omega * t_midpoint)
        y_intermediate = y_prev + (t_step / 2.0) * slope_at_t_mid_using_y_prev

        # Stage 2: Calculate y at t_current_step_end
        # Uses slope at t_current_step_end, with y_intermediate
        slope_at_t_end_using_y_intermediate = -alpha * y_intermediate + np.sin(omega * t_current_step_end)
        y_next = y_prev + t_step * slope_at_t_end_using_y_intermediate
        y.append(y_next)

    t_points_tensor = torch.tensor(t_points, dtype=torch.float32).view(-1, 1)
    y_values_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return t_points_tensor, y_values_tensor
