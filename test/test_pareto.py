import numpy as np
from spotpython.mo.pareto import is_pareto_efficient

def test_is_pareto_efficient_minimize():
    costs = np.array([[1, 2], [2, 1], [3, 3], [1.5, 1.5]])
    expected = np.array([True, True, False, True])
    result = is_pareto_efficient(costs, minimize=True)
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_is_pareto_efficient_maximize():
    costs = np.array([[1, 2], [2, 1], [3, 3], [1.5, 1.5]])
    expected = np.array([False, False, True, False])
    result = is_pareto_efficient(costs, minimize=False)
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_is_pareto_efficient_single_point():
    costs = np.array([[1, 2]])
    expected = np.array([True])
    result = is_pareto_efficient(costs, minimize=True)
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_is_pareto_efficient_identical_points():
    costs = np.array([[1, 2], [1, 2], [1, 2]])
    expected = np.array([True, False, False])
    result = is_pareto_efficient(costs, minimize=True)
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_is_pareto_efficient_empty_input():
    costs = np.array([])
    expected = np.array([])
    result = is_pareto_efficient(costs, minimize=True)
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"