import numpy as np
from spotpython.utils.aggregate import select_distant_points

def test_select_distant_points():
    # Test case 1: Basic example from the docstring
    X1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y1 = np.array([1, 2, 3, 4, 5])
    k1 = 3
    expected_selected_points1 = np.array([[5, 6], [9, 10], [1, 2]])
    expected_selected_y1 = np.array([3, 5, 1])
    selected_points1, selected_y1 = select_distant_points(X1, y1, k1)
    np.testing.assert_array_equal(np.sort(selected_points1, axis=0), np.sort(expected_selected_points1, axis=0),
                                  err_msg="Test case 1 failed: Selected points do not match expected values (order-independent)")
    np.testing.assert_array_equal(np.sort(selected_y1), np.sort(expected_selected_y1),
                                  err_msg="Test case 1 failed: Selected y values do not match expected values (order-independent)")

    # Test case 2: Fewer points than clusters requested
    X2 = np.array([[1, 2], [3, 4]])
    y2 = np.array([1, 2])
    k2 = 3
    # In this scenario, we expect an error because k can't be greater than the number of points
    try:
        select_distant_points(X2, y2, k2)
        assert False, "Test case 2 failed: The function should raise an error when k > n"
    except ValueError as e:
        assert str(e) == "n_samples=2 should be >= n_clusters=3.", f"Unexpected error message: {str(e)}"
    
    # Test case 3: k equals number of points
    X3 = np.array([[1, 1], [2, 2], [3, 3]])
    y3 = np.array([10, 20, 30])
    k3 = 3
    expected_selected_points3 = X3
    expected_selected_y3 = y3
    selected_points3, selected_y3 = select_distant_points(X3, y3, k3)
    np.testing.assert_array_equal(np.sort(selected_points3, axis=0), np.sort(expected_selected_points3, axis=0),
                                  err_msg="Test case 3 failed: Selected points do not match expected values (order-independent)")
    np.testing.assert_array_equal(np.sort(selected_y3), np.sort(expected_selected_y3),
                                  err_msg="Test case 3 failed: Selected y values do not match expected values (order-independent)")

    # Test case 4: k of 1 (should return the single most central point)
    X4 = np.array([[1, 2], [3, 4], [5, 6]])
    y4 = np.array([7, 8, 9])
    k4 = 1
    expected_selected_points4 = np.array([[3, 4]])  # This is the most central point
    expected_selected_y4 = np.array([8])
    selected_points4, selected_y4 = select_distant_points(X4, y4, k4)
    np.testing.assert_array_equal(np.sort(selected_points4, axis=0), np.sort(expected_selected_points4, axis=0),
                                  err_msg="Test case 4 failed: Selected points do not match expected values (order-independent)")
    np.testing.assert_array_equal(np.sort(selected_y4), np.sort(expected_selected_y4),
                                  err_msg="Test case 4 failed: Selected y values do not match expected values (order-independent)")

    # Test case 5: Large number of points with lower k
    X5 = np.random.random((100, 2))  # Random data points in 2D
    y5 = np.arange(100)  # Simple sequential values
    k5 = 5
    selected_points5, selected_y5 = select_distant_points(X5, y5, k5)
    assert selected_points5.shape == (k5, 2), "Test case 5 failed: Incorrect shape for selected X values"
    assert selected_y5.shape == (k5,), "Test case 5 failed: Incorrect shape for selected y values"