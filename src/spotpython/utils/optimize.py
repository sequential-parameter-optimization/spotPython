import numpy as np
from scipy.optimize import minimize


def run_minimize_with_restarts(objective, gradient, x0, bounds, n_restarts_optimizer=5, method="L-BFGS-B", maxit=100, verb=0, random_state=None) -> "minimize":
    """
    Runs multiple restarts of the minimize() function and returns the best found result.

    Args:
        objective (callable): The objective function to minimize.
        gradient (callable): The gradient of the objective.
        x0 (np.ndarray): Initial guess for the optimizer.
        bounds (list): List of (min, max) pairs for each element in x0.
        n_restarts_optimizer (int): Number of random-restart attempts.
        method (str): Optimization method. Default "L-BFGS-B".
        maxit (int): Max iterations.
        verb (int): Verbosity level.
        random_state (int, optional): Seed for the random-number generator to ensure reproducibility.

    Returns:
        OptimizeResult (object): The best optimization result among all restarts,
        represented as a ``OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if the optimizer
        exited successfully and ``message`` which describes the cause of the termination.
    """
    if random_state is not None:
        np.random.seed(random_state)

    best_result = None
    best_fun = float("inf")

    for _ in range(n_restarts_optimizer):
        # Create a random starting point within bounds
        x0_rand = []
        for (lb, ub), init_val in zip(bounds, x0):
            if lb == -np.inf or ub == np.inf:
                # If unbounded, keep the same initial guess
                x0_rand.append(init_val)
            else:
                x0_rand.append(np.random.uniform(lb, ub))
        x0_rand = np.array(x0_rand)

        result = minimize(
            fun=objective,
            x0=x0_rand,
            method=method,
            jac=gradient,
            bounds=bounds,
            options={"maxiter": maxit, "disp": verb > 0},
        )
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result

    return best_result
