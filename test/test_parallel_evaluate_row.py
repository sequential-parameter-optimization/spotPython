import numpy as np
from spotpython.utils.parallel import evaluate_row

def sample_objective(row, control):
    return row + control.get('offset', 0)

def test_evaluate_row_with_list():
    row = [1, 2, 3]
    fun_control = {'offset': 10}
    result = evaluate_row(row, sample_objective, fun_control)
    result = np.squeeze(result)  # Remove the extra dimension
    expected_result = np.array([11, 12, 13])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_evaluate_row_with_ndarray():
    row = np.array([1, 2, 3])
    fun_control = {'offset': 10}
    result = evaluate_row(row, sample_objective, fun_control)
    result = np.squeeze(result)  # Remove the extra dimension
    expected_result = np.array([11, 12, 13])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_evaluate_row_without_offset():
    row = np.array([4, 5, 6])
    fun_control = {}
    result = evaluate_row(row, sample_objective, fun_control)
    result = np.squeeze(result)  # Remove the extra dimension
    expected_result = np.array([4, 5, 6])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_evaluate_row_with_different_offset():
    row = np.array([7, 8, 9])
    fun_control = {'offset': -5}
    result = evaluate_row(row, sample_objective, fun_control)
    result = np.squeeze(result)  # Remove the extra dimension
    expected_result = np.array([2, 3, 4])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_evaluate_row_with_float_values():
    row = np.array([1.5, 2.5, 3.5])
    fun_control = {'offset': 10}
    result = evaluate_row(row, sample_objective, fun_control)
    result = np.squeeze(result)  # Remove the extra dimension
    expected_result = np.array([11.5, 12.5, 13.5])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_evaluate_row_with_negative_values():
    row = np.array([-1, -2, -3])
    fun_control = {'offset': 10}
    result = evaluate_row(row, sample_objective, fun_control)
    result = np.squeeze(result)  # Remove the extra dimension
    expected_result = np.array([9, 8, 7])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_evaluate_row_with_zero_values():
    row = np.array([0, 0, 0])
    fun_control = {'offset': 10}
    result = evaluate_row(row, sample_objective, fun_control)
    result = np.squeeze(result)  # Remove the extra dimension
    expected_result = np.array([10, 10, 10])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_evaluate_row_with_empty_row():
    row = np.array([])
    fun_control = {'offset': 10}
    result = evaluate_row(row, sample_objective, fun_control)
    result = np.squeeze(result)  # Remove the extra dimension
    expected_result = np.array([])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"