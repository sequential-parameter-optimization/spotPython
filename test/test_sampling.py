# import pytest
# import numpy as np
# from spotpython.utils.sampling import fullfactorial, rlh, jd, mm, mmphi, mmsort, perturb, mmlhs, bestlh, phisort, subset

# def test_fullfactorial():
#     """
#     Test the fullfactorial function with various scenarios.
#     """

#     # Test case 1: Basic functionality with Edges=1 (points equally spaced from edge to edge)
#     q = [2, 3]
#     expected_output = np.array([
#         [0.0, 0.0],
#         [0.0, 0.5],
#         [0.0, 1.0],
#         [1.0, 0.0],
#         [1.0, 0.5],
#         [1.0, 1.0]
#     ])
#     np.testing.assert_almost_equal(fullfactorial(q, Edges=1), expected_output, decimal=7)

#     # Test case 2: Basic functionality with Edges=0 (points at midpoints of bins)
#     # In the first dimension with q[0] = 2, the midpoints are [0.0, 0.75].
#     # In the second dimension with q[1] = 3, the midpoints are [0.0, 0.4167, 0.8333].
#     actual_output = fullfactorial(q, Edges=0)
#     expected_output_edges0 = np.array([
#         [0.0, 0.0],
#         [0.0, 0.4166667],
#         [0.0, 0.8333333],
#         [0.75, 0.0],
#         [0.75, 0.4166667],
#         [0.75, 0.8333333]
#     ])
#     np.testing.assert_almost_equal(actual_output, expected_output_edges0, decimal=7)

#     # Test case 3: Check if ValueError is raised for dimensions with less than 2 points
#     q_invalid = [1, 3]
#     with pytest.raises(ValueError):
#         fullfactorial(q_invalid, Edges=1)

#     # Test case 4: Check with a single dimension and multiple levels
#     q_single = [5]
#     expected_output_single = np.linspace(0, 1, 5).reshape((5, 1))
#     np.testing.assert_almost_equal(fullfactorial(q_single, Edges=1), expected_output_single, decimal=7)

#     # Test case 5: Verify shape for higher dimensions with Edges=1
#     q_higher = [5, 2]
#     output = fullfactorial(q_higher, Edges=1)
#     assert output.shape == (10, 2), f"Expected shape (10, 2), got {output.shape}"

# def test_rlh():
#     """
#     Test the rlh function with various scenarios.
#     """

#     # Test case 1: Basic functionality with edges=0
#     n, k = 5, 2
#     output = rlh(n, k, edges=0)
#     assert output.shape == (n, k), f"Expected shape ({n}, {k}), got {output.shape}"
#     assert np.all(output >= 0) and np.all(output <= 1), "All values should be within [0, 1]"
#     for i in range(k):
#         assert len(np.unique(output[:, i])) == n, f"Column {i} should have {n} unique values"

#     # Test case 2: Basic functionality with edges=1
#     output_edges = rlh(n, k, edges=1)
#     assert output_edges.shape == (n, k), f"Expected shape ({n}, {k}), got {output_edges.shape}"
#     assert np.all(output_edges >= 0) and np.all(output_edges <= 1), "All values should be within [0, 1]"
#     for i in range(k):
#         assert len(np.unique(output_edges[:, i])) == n, f"Column {i} should have {n} unique values"

#     # Test case 3: Check for single dimension (k=1)
#     n, k = 10, 1
#     output_single_dim = rlh(n, k, edges=0)
#     assert output_single_dim.shape == (n, k), f"Expected shape ({n}, {k}), got {output_single_dim.shape}"
#     assert np.all(output_single_dim >= 0) and np.all(output_single_dim <= 1), "All values should be within [0, 1]"
#     assert len(np.unique(output_single_dim[:, 0])) == n, "Column 0 should have n unique values"

#     # Test case 4: Check for single point (n=1)
#     n, k = 1, 3
#     output_single_point = rlh(n, k, edges=0)
#     assert output_single_point.shape == (n, k), f"Expected shape ({n}, {k}), got {output_single_point.shape}"
#     assert np.all(output_single_point >= 0) and np.all(output_single_point <= 1), "All values should be within [0, 1]"

#     # Test case 5: Check for higher dimensions
#     n, k = 7, 5
#     output_higher_dim = rlh(n, k, edges=0)
#     assert output_higher_dim.shape == (n, k), f"Expected shape ({n}, {k}), got {output_higher_dim.shape}"
#     assert np.all(output_higher_dim >= 0) and np.all(output_higher_dim <= 1), "All values should be within [0, 1]"
#     for i in range(k):
#         assert len(np.unique(output_higher_dim[:, i])) == n, f"Column {i} should have {n} unique values"

#     # Test case 6: Check for edges=1 scaling
#     n, k = 4, 2
#     output_edges_check = rlh(n, k, edges=1)
#     assert np.isclose(output_edges_check.min(), 0.0), "Minimum value should be close to 0.0 for edges=1"
#     assert np.isclose(output_edges_check.max(), 1.0), "Maximum value should be close to 1.0 for edges=1"
        
# def test_jd():
#     """
#     Test the jd function with various scenarios.
#     """

#     # Test case 1: Basic functionality with Euclidean norm (p=2)
#     X = np.array([[0.0, 0.0],
#                     [1.0, 1.0],
#                     [2.0, 2.0]])
#     J, distinct_d = jd(X, p=2.0)
#     expected_distances = np.array([np.sqrt(2), 2 * np.sqrt(2)])
#     expected_counts = np.array([2, 1])  # sqrt(2) occurs twice, 2*sqrt(2) occurs once
#     np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
#     np.testing.assert_array_equal(J, expected_counts)

#     # Test case 2: Basic functionality with Manhattan norm (p=1)
#     J, distinct_d = jd(X, p=1.0)
#     expected_distances = np.array([2.0, 4.0])  # Manhattan distances
#     expected_counts = np.array([2, 1])  # 2.0 occurs twice, 4.0 occurs once
#     np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
#     np.testing.assert_array_equal(J, expected_counts)

#     # Test case 3: Single point (no distances)
#     X_single = np.array([[0.0, 0.0]])
#     J, distinct_d = jd(X_single, p=2.0)
#     assert len(distinct_d) == 0, "There should be no distances for a single point"
#     assert len(J) == 0, "There should be no multiplicities for a single point"

#     # Test case 4: Two points
#     X_two = np.array([[0.0, 0.0],
#                         [3.0, 4.0]])
#     J, distinct_d = jd(X_two, p=2.0)
#     expected_distances = np.array([5.0])  # Euclidean distance
#     expected_counts = np.array([1])  # Only one distance
#     np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
#     np.testing.assert_array_equal(J, expected_counts)

#     # Test case 5: Higher dimensions
#     X_high_dim = np.array([[0.0, 0.0, 0.0],
#                             [1.0, 1.0, 1.0],
#                             [2.0, 2.0, 2.0]])
#     J, distinct_d = jd(X_high_dim, p=2.0)
#     expected_distances = np.array([np.sqrt(3), 2 * np.sqrt(3)])
#     expected_counts = np.array([2, 1])  # sqrt(3) occurs twice, 2*sqrt(3) occurs once
#     np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
#     np.testing.assert_array_equal(J, expected_counts)

#     # Test case 6: Check with p=inf (Chebyshev distance)
#     X_inf = np.array([[0.0, 0.0],
#                       [1.0, 3.0],
#                       [4.0, 1.0]])
#     J, distinct_d = jd(X_inf, p=np.inf)
#     # Correct distances: [3, 4, 3] => distinct_d = [3, 4], multiplicities J = [2, 1]
#     expected_distances = np.array([3.0, 4.0])
#     expected_counts = np.array([2, 1])
#     np.testing.assert_almost_equal(distinct_d, expected_distances, decimal=7)
#     np.testing.assert_array_equal(J, expected_counts)

# def test_mm():
#     """
#     Test the mm function with various scenarios.
#     """

#     # Test case 1: Identical sampling plans
#     X1 = np.array([[0.0, 0.0],
#                     [0.5, 0.5],
#                     [1.0, 1.0]])
#     X2 = np.array([[0.0, 0.0],
#                     [0.5, 0.5],
#                     [1.0, 1.0]])
#     assert mm(X1, X2, p=2.0) == 0, "Identical sampling plans should return 0"

#     # Test case 2: X1 is more space-filling than X2
#     X1 = np.array([[0.0, 0.0],
#                     [0.5, 0.5],
#                     [1.0, 1.0]])
#     X2 = np.array([[0.0, 0.0],
#                     [0.1, 0.1],
#                     [0.2, 0.2]])
#     assert mm(X1, X2, p=2.0) == 1, "X1 should be more space-filling than X2"

#     # Test case 3: X2 is more space-filling than X1
#     X1 = np.array([[0.0, 0.0],
#                     [0.1, 0.1],
#                     [0.2, 0.2]])
#     X2 = np.array([[0.0, 0.0],
#                     [0.5, 0.5],
#                     [1.0, 1.0]])
#     assert mm(X1, X2, p=2.0) == 2, "X2 should be more space-filling than X1"

#     # Test case 4: Higher dimensions with p=1 (Manhattan distance)
#     X1 = np.array([[0.0, 0.0, 0.0],
#                     [0.5, 0.5, 0.5],
#                     [1.0, 1.0, 1.0]])
#     X2 = np.array([[0.0, 0.0, 0.0],
#                     [0.1, 0.1, 0.1],
#                     [0.2, 0.2, 0.2]])
#     assert mm(X1, X2, p=1.0) == 1, "X1 should be more space-filling than X2 with Manhattan distance"

#     # Test case 5: Single point in each sampling plan
#     X1 = np.array([[0.0, 0.0]])
#     X2 = np.array([[1.0, 1.0]])
#     assert mm(X1, X2, p=2.0) == 0, "Single points should be considered equally space-filling"

#     # Test case 6: Edge case with empty sampling plans
#     X1 = np.empty((0, 2))
#     X2 = np.empty((0, 2))
#     assert mm(X1, X2, p=2.0) == 0, "Empty sampling plans should return 0"

# def test_mmphi():
#     """
#     Test the mmphi function with various scenarios.
#     """

#     # Test case 1: Basic functionality with q=2 and p=2 (Euclidean distance)
#     X = np.array([
#         [0.0, 0.0],
#         [0.5, 0.5],
#         [1.0, 1.0]
#     ])
#     quality = mmphi(X, q=2, p=2)
#     assert quality > 0, "Quality metric should be positive for valid input"

#     # Test case 2: Single point (no distances)
#     X_single = np.array([[0.0, 0.0]])
#     quality_single = mmphi(X_single, q=2, p=2)
#     assert quality_single == 0, "Quality metric should be 0 for a single point"

#     # Test case 3: Two points
#     X_two = np.array([
#         [0.0, 0.0],
#         [1.0, 1.0]
#     ])
#     quality_two = mmphi(X_two, q=2, p=2)
#     assert quality_two > 0, "Quality metric should be positive for two points"

#     # Test case 4: Higher dimensions
#     X_high_dim = np.array([
#         [0.0, 0.0, 0.0],
#         [0.5, 0.5, 0.5],
#         [1.0, 1.0, 1.0]
#     ])
#     quality_high_dim = mmphi(X_high_dim, q=2, p=2)
#     assert quality_high_dim > 0, "Quality metric should be positive for higher dimensions"

#     # Test case 5: Check with Manhattan distance (p=1)
#     X_manhattan = np.array([
#         [0.0, 0.0],
#         [0.5, 0.5],
#         [1.0, 1.0]
#     ])
#     quality_manhattan = mmphi(X_manhattan, q=2, p=1)
#     assert quality_manhattan > 0, "Quality metric should be positive with Manhattan distance"

#     # Test case 6: Edge case with empty sampling plan
#     X_empty = np.empty((0, 2))
#     quality_empty = mmphi(X_empty, q=2, p=2)
#     assert quality_empty == 0, "Quality metric should be 0 for an empty sampling plan"

#     # Test case 7: Check with q=1 (different exponent)
#     X_q1 = np.array([
#         [0.0, 0.0],
#         [0.5, 0.5],
#         [1.0, 1.0]
#     ])
#     quality_q1 = mmphi(X_q1, q=1, p=2)
#     assert quality_q1 > 0, "Quality metric should be positive with q=1"

# def test_mmsort():
#     """
#     Test the mmsort function with various scenarios.
#     """

#     # Test case 1: Two sampling plans in 2D
#     X1 = np.array([[0.0, 0.0],
#                     [0.5, 0.5],
#                     [1.0, 1.0]])
#     X2 = np.array([[0.2, 0.2],
#                     [0.6, 0.4],
#                     [0.9, 0.9]])
#     X3D = np.stack([X1, X2], axis=2)
#     ranking = mmsort(X3D, p=2.0)
#     assert len(ranking) == 2, "Ranking should have 2 elements for 2 sampling plans"
#     assert set(ranking) == {1, 2}, "Ranking should contain indices 1 and 2"

#     # Test case 2: Three sampling plans in 2D
#     X3 = np.array([[0.1, 0.1],
#                     [0.4, 0.4],
#                     [0.8, 0.8]])
#     X3D = np.stack([X1, X2, X3], axis=2)
#     ranking = mmsort(X3D, p=2.0)
#     assert len(ranking) == 3, "Ranking should have 3 elements for 3 sampling plans"
#     assert set(ranking) == {1, 2, 3}, "Ranking should contain indices 1, 2, and 3"

#     # Test case 3: Single sampling plan
#     X3D = np.expand_dims(X1, axis=2)
#     ranking = mmsort(X3D, p=2.0)
#     assert len(ranking) == 1, "Ranking should have 1 element for a single sampling plan"
#     assert ranking[0] == 1, "Ranking should be [1] for a single sampling plan"

#     # Test case 4: Higher dimensions
#     X1 = np.array([[0.0, 0.0, 0.0],
#                     [0.5, 0.5, 0.5],
#                     [1.0, 1.0, 1.0]])
#     X2 = np.array([[0.1, 0.1, 0.1],
#                     [0.4, 0.4, 0.4],
#                     [0.9, 0.9, 0.9]])
#     X3D = np.stack([X1, X2], axis=2)
#     ranking = mmsort(X3D, p=2.0)
#     assert len(ranking) == 2, "Ranking should have 2 elements for 2 sampling plans in higher dimensions"
#     assert set(ranking) == {1, 2}, "Ranking should contain indices 1 and 2"

#     # Test case 5: Edge case with empty sampling plans
#     X3D = np.empty((0, 2, 2))
#     ranking = mmsort(X3D, p=2.0)
#     assert len(ranking) == 2, "Ranking should handle empty sampling plans and return indices"

#     # Test case 6: Edge case with identical sampling plans
#     X3D = np.stack([X1, X1], axis=2)
#     ranking = mmsort(X3D, p=2.0)
#     assert len(ranking) == 2, "Ranking should have 2 elements for identical sampling plans"
#     assert set(ranking) == {1, 2}, "Ranking should contain indices 1 and 2 for identical plans"

# def test_perturb():
#     """
#     Test the perturb function with various scenarios.
#     """

#     # Test case 1: Basic functionality with a single perturbation
#     X_original = np.array([
#         [1, 3],
#         [2, 4],
#         [3, 1],
#         [4, 2]
#     ])
#     X_perturbed = perturb(X_original.copy(), PertNum=1)
#     assert X_perturbed.shape == X_original.shape, "Shape of perturbed array should match the original"
#     for col in range(X_original.shape[1]):
#         assert set(X_perturbed[:, col]) == set(X_original[:, col]), f"Column {col} should remain a permutation"

#     # Test case 2: Multiple perturbations
#     X_original = np.array([
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ])
#     X_perturbed = perturb(X_original.copy(), PertNum=5)
#     assert X_perturbed.shape == X_original.shape, "Shape of perturbed array should match the original"
#     for col in range(X_original.shape[1]):
#         assert set(X_perturbed[:, col]) == set(X_original[:, col]), f"Column {col} should remain a permutation"

#     # Test case 3: No perturbations (PertNum=0)
#     X_original = np.array([
#         [1, 2],
#         [3, 4]
#     ])
#     X_perturbed = perturb(X_original.copy(), PertNum=0)
#     np.testing.assert_array_equal(X_perturbed, X_original, "No perturbations should result in identical array")

#     # Test case 4: Single column (k=1)
#     X_original = np.array([
#         [1],
#         [2],
#         [3],
#         [4]
#     ])
#     X_perturbed = perturb(X_original.copy(), PertNum=2)
#     assert X_perturbed.shape == X_original.shape, "Shape of perturbed array should match the original"
#     assert set(X_perturbed[:, 0]) == set(X_original[:, 0]), "Single column should remain a permutation"

#     # Test case 5: Single row (n=1)
#     X_original = np.array([[1, 2, 3]])
#     X_perturbed = perturb(X_original.copy(), PertNum=3)
#     np.testing.assert_array_equal(X_perturbed, X_original, "Single row should remain unchanged")

#     # Test case 6: Edge case with empty array
#     X_original = np.empty((0, 2))
#     X_perturbed = perturb(X_original.copy(), PertNum=1)
#     np.testing.assert_array_equal(X_perturbed, X_original, "Empty array should remain unchanged")
    
# def test_mmlhs():
#     """
#     Test the mmlhs function with various scenarios.
#     """

#     # Test case 1: Basic functionality with small population and iterations
#     X_start = np.array([
#         [0, 0],
#         [1, 3],
#         [2, 1],
#         [3, 2]
#     ])
#     population = 5
#     iterations = 10
#     X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)
#     assert X_opt.shape == X_start.shape, "Optimized plan should have the same shape as the initial plan"
#     for col in range(X_start.shape[1]):
#         assert set(X_opt[:, col]) == set(X_start[:, col]), f"Column {col} should remain a permutation"

#     # Test case 2: Single point (n=1)
#     X_start = np.array([[0]])
#     population = 3
#     iterations = 5
#     X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)
#     np.testing.assert_array_equal(X_opt, X_start, "Single point should remain unchanged")

#     # Test case 3: Single dimension (k=1)
#     X_start = np.array([
#         [0],
#         [1],
#         [2],
#         [3]
#     ])
#     population = 4
#     iterations = 15
#     X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)
#     assert X_opt.shape == X_start.shape, "Optimized plan should have the same shape as the initial plan"
#     assert set(X_opt[:, 0]) == set(X_start[:, 0]), "Single column should remain a permutation"

#     # Test case 4: Higher dimensions
#     X_start = np.array([
#         [0, 0, 0],
#         [1, 1, 1],
#         [2, 2, 2],
#         [3, 3, 3]
#     ])
#     population = 10
#     iterations = 20
#     X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)
#     assert X_opt.shape == X_start.shape, "Optimized plan should have the same shape as the initial plan"
#     for col in range(X_start.shape[1]):
#         assert set(X_opt[:, col]) == set(X_start[:, col]), f"Column {col} should remain a permutation"

#     # Test case 5: Edge case with empty array
#     X_start = np.empty((0, 2))
#     population = 5
#     iterations = 10
#     X_opt = mmlhs(X_start, population=population, iterations=iterations, q=2.0)
#     np.testing.assert_array_equal(X_opt, X_start, "Empty array should remain unchanged")

#     # Test case 6: Check with different q values
#     X_start = np.array([
#         [0, 0],
#         [1, 3],
#         [2, 1],
#         [3, 2]
#     ])
#     population = 5
#     iterations = 10
#     X_opt_q1 = mmlhs(X_start, population=population, iterations=iterations, q=1.0)
#     X_opt_q3 = mmlhs(X_start, population=population, iterations=iterations, q=3.0)
#     assert X_opt_q1.shape == X_start.shape, "Optimized plan with q=1 should have the same shape as the initial plan"
#     assert X_opt_q3.shape == X_start.shape, "Optimized plan with q=3 should have the same shape as the initial plan"
#     for col in range(X_start.shape[1]):
#         assert set(X_opt_q1[:, col]) == set(X_start[:, col]), f"Column {col} should remain a permutation for q=1"
#         assert set(X_opt_q3[:, col]) == set(X_start[:, col]), f"Column {col} should remain a permutation for q=3"

# def test_bestlh():
#     """
#     Test the bestlh function with various scenarios.
#     """

#     # Test case 1: Basic functionality with small population and iterations
#     n, k, population, iterations = 5, 2, 5, 10
#     X_opt = bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)
#     assert X_opt.shape == (n, k), "Optimized plan should have the correct shape"
#     for col in range(k):
#         assert len(np.unique(X_opt[:, col])) == n, f"Column {col} should have {n} unique values"

#     # Test case 2: Single dimension (k=2 is the minimum allowed)
#     n, k, population, iterations = 4, 2, 5, 10
#     X_opt = bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)
#     assert X_opt.shape == (n, k), "Optimized plan should have the correct shape"
#     for col in range(k):
#         assert len(np.unique(X_opt[:, col])) == n, f"Column {col} should have {n} unique values"

#     # Test case 3: Higher dimensions
#     n, k, population, iterations = 7, 5, 10, 20
#     X_opt = bestlh(n=n, k=k, population=population, iterations=iterations, p=2, plot=False)
#     assert X_opt.shape == (n, k), "Optimized plan should have the correct shape"
#     for col in range(k):
#         assert len(np.unique(X_opt[:, col])) == n, f"Column {col} should have {n} unique values"

#     # Test case 4: Edge case with minimum points (n=2)
#     n, k, population, iterations = 2, 2, 5, 10
#     X_opt = bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)
#     assert X_opt.shape == (n, k), "Optimized plan should have the correct shape"
#     for col in range(k):
#         assert len(np.unique(X_opt[:, col])) == n, f"Column {col} should have {n} unique values"

#     # Test case 5: Check with different distance norms (p=1 and p=2)
#     n, k, population, iterations = 6, 3, 5, 15
#     X_opt_p1 = bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)
#     X_opt_p2 = bestlh(n=n, k=k, population=population, iterations=iterations, p=2, plot=False)
#     assert X_opt_p1.shape == (n, k), "Optimized plan with p=1 should have the correct shape"
#     assert X_opt_p2.shape == (n, k), "Optimized plan with p=2 should have the correct shape"

#     # Test case 6: Check if ValueError is raised for k < 2
#     n, k, population, iterations = 5, 1, 5, 10
#     with pytest.raises(ValueError):
#         bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)

#     # Test case 7: Edge case with empty Latin hypercube
#     n, k, population, iterations = 0, 2, 5, 10
#     with pytest.raises(ValueError):
#         bestlh(n=n, k=k, population=population, iterations=iterations, p=1, plot=False)

# def test_phisort():
#     """
#     Test the phisort function with various scenarios.
#     """

#     # Test case 1: Two sampling plans in 2D
#     X1 = np.array([[0.0, 0.0],
#                     [0.5, 0.5],
#                     [1.0, 1.0]])
#     X2 = np.array([[0.2, 0.2],
#                     [0.6, 0.4],
#                     [0.9, 0.9]])
#     X3D = np.stack([X1, X2], axis=2)
#     ranking = phisort(X3D, q=2.0, p=2.0)
#     assert len(ranking) == 2, "Ranking should have 2 elements for 2 sampling plans"
#     assert set(ranking) == {1, 2}, "Ranking should contain indices 1 and 2"

#     # Test case 2: Three sampling plans in 2D
#     X3 = np.array([[0.1, 0.1],
#                     [0.4, 0.4],
#                     [0.8, 0.8]])
#     X3D = np.stack([X1, X2, X3], axis=2)
#     ranking = phisort(X3D, q=2.0, p=2.0)
#     assert len(ranking) == 3, "Ranking should have 3 elements for 3 sampling plans"
#     assert set(ranking) == {1, 2, 3}, "Ranking should contain indices 1, 2, and 3"

#     # Test case 3: Single sampling plan
#     X3D = np.expand_dims(X1, axis=2)
#     ranking = phisort(X3D, q=2.0, p=2.0)
#     assert len(ranking) == 1, "Ranking should have 1 element for a single sampling plan"
#     assert ranking[0] == 1, "Ranking should be [1] for a single sampling plan"

#     # Test case 4: Higher dimensions
#     X1 = np.array([[0.0, 0.0, 0.0],
#                     [0.5, 0.5, 0.5],
#                     [1.0, 1.0, 1.0]])
#     X2 = np.array([[0.1, 0.1, 0.1],
#                     [0.4, 0.4, 0.4],
#                     [0.9, 0.9, 0.9]])
#     X3D = np.stack([X1, X2], axis=2)
#     ranking = phisort(X3D, q=2.0, p=2.0)
#     assert len(ranking) == 2, "Ranking should have 2 elements for 2 sampling plans in higher dimensions"
#     assert set(ranking) == {1, 2}, "Ranking should contain indices 1 and 2"

#     # Test case 5: Edge case with empty sampling plans
#     X3D = np.empty((0, 2, 2))
#     ranking = phisort(X3D, q=2.0, p=2.0)
#     assert len(ranking) == 2, "Ranking should handle empty sampling plans and return indices"

#     # Test case 6: Edge case with identical sampling plans
#     X3D = np.stack([X1, X1], axis=2)
#     ranking = phisort(X3D, q=2.0, p=2.0)
#     assert len(ranking) == 2, "Ranking should have 2 elements for identical sampling plans"
#     assert set(ranking) == {1, 2}, "Ranking should contain indices 1 and 2 for identical plans"

#     # Test case 7: Check with different q values
#     X3D = np.stack([X1, X2], axis=2)
#     ranking_q1 = phisort(X3D, q=1.0, p=2.0)
#     ranking_q3 = phisort(X3D, q=3.0, p=2.0)
#     assert len(ranking_q1) == 2, "Ranking with q=1 should have 2 elements"
#     assert len(ranking_q3) == 2, "Ranking with q=3 should have 2 elements"
#     assert set(ranking_q1) == {1, 2}, "Ranking with q=1 should contain indices 1 and 2"
#     assert set(ranking_q3) == {1, 2}, "Ranking with q=3 should contain indices 1 and 2"

# def test_subset():
#     """
#     Test the subset function with various scenarios.
#     """

#     # Test case 1: Basic functionality with a 5-point plan in 2D
#     X = np.array([
#         [0.0, 0.0],
#         [0.0, 1.0],
#         [1.0, 0.0],
#         [1.0, 1.0],
#         [0.5, 0.5]
#     ])
#     ns = 3
#     Xs, Xr = subset(X, ns=ns)
#     assert Xs.shape == (ns, X.shape[1]), f"Subset should have shape ({ns}, {X.shape[1]})"
#     assert Xr.shape == (X.shape[0] - ns, X.shape[1]), f"Remainder should have shape ({X.shape[0] - ns}, {X.shape[1]})"
#     assert np.all(np.isin(Xs, X)), "Subset points should be part of the original sampling plan"
#     assert np.all(np.isin(Xr, X)), "Remainder points should be part of the original sampling plan"
#     assert len(np.unique(np.vstack((Xs, Xr)), axis=0)) == X.shape[0], "Subset and remainder should not overlap"

#     # Test case 2: Subset size equal to the total number of points
#     ns = X.shape[0]
#     Xs, Xr = subset(X, ns=ns)
#     assert Xs.shape == (ns, X.shape[1]), f"Subset should have shape ({ns}, {X.shape[1]})"
#     assert Xr.shape == (0, X.shape[1]), "Remainder should be empty when subset size equals total points"
#     np.testing.assert_array_equal(np.sort(Xs, axis=0), np.sort(X, axis=0), "Subset should contain all points")

#     # Test case 3: Subset size of 1
#     ns = 1
#     Xs, Xr = subset(X, ns=ns)
#     assert Xs.shape == (ns, X.shape[1]), f"Subset should have shape ({ns}, {X.shape[1]})"
#     assert Xr.shape == (X.shape[0] - ns, X.shape[1]), f"Remainder should have shape ({X.shape[0] - ns}, {X.shape[1]})"
#     assert np.all(np.isin(Xs, X)), "Subset points should be part of the original sampling plan"
#     assert np.all(np.isin(Xr, X)), "Remainder points should be part of the original sampling plan"

#     # Test case 4: Higher dimensions
#     X_high_dim = np.array([
#         [0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0],
#         [2.0, 2.0, 2.0],
#         [3.0, 3.0, 3.0],
#         [4.0, 4.0, 4.0]
#     ])
#     ns = 2
#     Xs, Xr = subset(X_high_dim, ns=ns)
#     assert Xs.shape == (ns, X_high_dim.shape[1]), f"Subset should have shape ({ns}, {X_high_dim.shape[1]})"
#     assert Xr.shape == (X_high_dim.shape[0] - ns, X_high_dim.shape[1]), f"Remainder should have shape ({X_high_dim.shape[0] - ns}, {X_high_dim.shape[1]})"
#     assert np.all(np.isin(Xs, X_high_dim)), "Subset points should be part of the original sampling plan"
#     assert np.all(np.isin(Xr, X_high_dim)), "Remainder points should be part of the original sampling plan"

#     # Test case 5: Edge case with empty sampling plan
#     X_empty = np.empty((0, 2))
#     ns = 0
#     Xs, Xr = subset(X_empty, ns=ns)
#     assert Xs.shape == (0, 2), "Subset should be empty for an empty sampling plan"
#     assert Xr.shape == (0, 2), "Remainder should be empty for an empty sampling plan"

#     # Test case 6: Subset size larger than the total number of points
#     ns = X.shape[0] + 1
#     with pytest.raises(IndexError):
#         subset(X, ns=ns)










