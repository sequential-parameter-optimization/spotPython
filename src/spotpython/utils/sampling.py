import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def fullfactorial(q, Edges=1) -> np.ndarray:
    """Generates a full factorial sampling plan in the unit cube.

    Args:
        q (list or np.ndarray):
            A list or array containing the number of points along each dimension (k-vector).
        Edges (int, optional):
            Determines spacing of points. If `Edges=1`, points are equally spaced from edge to edge (default).
            Otherwise, points will be in the centers of n = q[0]*q[1]*...*q[k-1] bins filling the unit cube.

    Returns:
        (np.ndarray): Full factorial sampling plan as an array of shape (n, k), where n is the total number of points and k is the number of dimensions.

    Raises:
        ValueError: If any dimension in `q` is less than 2.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> from spotpython.utils.sampling import fullfactorial
        >>> q = [3, 2]
        >>> X = fullfactorial(q, Edges=0)
        >>> print(X)
                [[0.         0.        ]
                [0.         0.75      ]
                [0.41666667 0.        ]
                [0.41666667 0.75      ]
                [0.83333333 0.        ]
                [0.83333333 0.75      ]]
        >>> X = fullfactorial(q, Edges=1)
        >>> print(X)
                [[0.  0. ]
                [0.  1. ]
                [0.5 0. ]
                [0.5 1. ]
                [1.  0. ]
                [1.  1. ]]

    """
    q = np.array(q)
    if np.min(q) < 2:
        raise ValueError("You must have at least two points per dimension.")

    # Total number of points in the sampling plan
    n = np.prod(q)

    # Number of dimensions
    k = len(q)

    # Pre-allocate memory for the sampling plan
    X = np.zeros((n, k))

    # Additional phantom element
    q = np.append(q, 1)

    for j in range(k):
        if Edges == 1:
            one_d_slice = np.linspace(0, 1, q[j])
        else:
            one_d_slice = np.linspace(1 / (2 * q[j]), 1, q[j]) - 1 / (2 * q[j])

        column = np.array([])

        while len(column) < n:
            for ll in range(q[j]):
                column = np.append(column, np.ones(np.prod(q[j + 1 : k])) * one_d_slice[ll])

        X[:, j] = column

    return X


def rlh(n: int, k: int, edges: int = 0) -> np.ndarray:
    """
    Generates a random Latin hypercube within the [0,1]^k hypercube.

    Args:
        n (int): Desired number of points.
        k (int): Number of design variables (dimensions).
        edges (int, optional):
            If 1, places centers of the extreme bins at the domain edges ([0,1]).
            Otherwise, bins are fully contained within the domain, i.e. midpoints.
            Defaults to 0.

    Returns:
        np.ndarray: A Latin hypercube sampling plan of n points in k dimensions,
                    with each coordinate in the range [0,1].

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.sampling import rlh
        >>> # Generate a 2D Latin hypercube with 5 points and edges=0
        >>> X = rlh(n=5, k=2, edges=0)
        >>> print(X)
        # Example output (values vary due to randomness):
        # [[0.1  0.5 ]
        #  [0.7  0.1 ]
        #  [0.9  0.7 ]
        #  [0.3  0.9 ]
        #  [0.5  0.3 ]]

    """
    # Initialize array
    X = np.zeros((n, k), dtype=float)

    # Fill with random permutations
    for i in range(k):
        X[:, i] = np.random.permutation(n)

    # Adjust normalization based on the edges flag
    if edges == 1:
        # [X=0..n-1] -> [0..1]
        X = X / (n - 1)
    else:
        # Points at true midpoints
        # [X=0..n-1] -> [0.5/n..(n-0.5)/n]
        X = (X + 0.5) / n

    return X


def jd(X: np.ndarray, p: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes and counts the distinct p-norm distances between all pairs of points in X.
    It returns:
    1) A list of distinct distances (sorted), and
    2) A corresponding multiplicity array that indicates how often each distance occurs.

    Args:
        X (np.ndarray):
            A 2D array of shape (n, d) representing n points in d-dimensional space.
        p (float, optional):
            The distance norm to use. p=1 uses the Manhattan (L1) norm, while p=2 uses the
            Euclidean (L2) norm. Defaults to 1.0 (Manhattan norm).

    Returns:
        (np.ndarray, np.ndarray):
            A tuple (J, distinct_d), where:
            - distinct_d is a 1D float array of unique, sorted distances between points.
            - J is a 1D integer array that provides the multiplicity (occurrence count)
              of each distance in distinct_d.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.sampling import jd
        >>> # A small 3-point set in 2D
        >>> X = np.array([[0.0, 0.0],
        ...               [1.0, 1.0],
        ...               [2.0, 2.0]])
        >>> J, distinct_d = jd(X, p=2.0)
        >>> print("Distinct distances:", distinct_d)
        >>> print("Occurrences:", J)
        # Possible output (using Euclidean norm):
        # Distinct distances: [1.41421356 2.82842712]
        # Occurrences: [1 1]
        # Explanation: Distances are sqrt(2) between consecutive points and 2*sqrt(2) for the farthest pair.
            Distinct distances: [1.41421356 2.82842712]
            Occurrences: [2 1]
    """
    n = X.shape[0]

    # Allocate enough space for all pairwise distances
    # (n*(n-1))/2 pairs for an n-point set
    pair_count = n * (n - 1) // 2
    d = np.zeros(pair_count, dtype=float)

    # Fill the distance array
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Compute the p-norm distance
            d[idx] = np.linalg.norm(X[i] - X[j], ord=p)
            idx += 1

    # Find unique distances and their multiplicities
    distinct_d = np.unique(d)
    J = np.zeros_like(distinct_d, dtype=int)
    for i, val in enumerate(distinct_d):
        J[i] = np.sum(d == val)

    return J, distinct_d


def mm(X1: np.ndarray, X2: np.ndarray, p: Optional[float] = 1.0) -> int:
    """
    Determines which of two sampling plans has better space-filling properties
    according to the Morris-Mitchell criterion.

    Args:
        X1 (np.ndarray): A 2D array representing the first sampling plan.
        X2 (np.ndarray): A 2D array representing the second sampling plan.
        p (float, optional): The distance metric. p=1 uses Manhattan (L1) distance,
            while p=2 uses Euclidean (L2). Defaults to 1.0.

    Returns:
        int:
            - 0 if both plans are identical or equally space-filling
            - 1 if X1 is more space-filling
            - 2 if X2 is more space-filling

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.sampling import mm
        >>> # Create two 3-point sampling plans in 2D
        >>> X1 = np.array([[0.0, 0.0],
        ...                [0.5, 0.5],
        ...                [0.0, 1.0]])
        >>> X2 = np.array([[0.1, 0.1],
        ...                [0.4, 0.6],
        ...                [0.1, 0.9]])
        >>> # Compare which plan has better space-filling (Morris-Mitchell)
        >>> better = mm(X1, X2, p=2.0)
        >>> print(better)
        # Prints either 0, 1, or 2 depending on which plan is more space-filling.
    """
    # Quick check if the sorted sets of points are identical
    # (mimicking MATLAB's sortrows check)
    X1_sorted = X1[np.lexsort(np.rot90(X1))]
    X2_sorted = X2[np.lexsort(np.rot90(X2))]
    if np.array_equal(X1_sorted, X2_sorted):
        return 0  # Identical sampling plans

    # Compute distance multiplicities for each plan
    J1, d1 = jd(X1, p)
    J2, d2 = jd(X2, p)
    m1, m2 = len(d1), len(d2)

    # Construct V1 and V2: alternate distance and negative multiplicity
    V1 = np.zeros(2 * m1)
    V1[0::2] = d1
    V1[1::2] = -J1

    V2 = np.zeros(2 * m2)
    V2[0::2] = d2
    V2[1::2] = -J2

    # Trim the longer vector to match the size of the shorter
    m = min(m1, m2)
    V1 = V1[:m]
    V2 = V2[:m]

    # Compare element-by-element:
    # c[i] = 1 if V1[i] > V2[i], 2 if V1[i] < V2[i], 0 otherwise.
    c = (V1 > V2).astype(int) + 2 * (V1 < V2).astype(int)

    if np.sum(c) == 0:
        # Equally space-filling
        return 0
    else:
        # The first non-zero entry indicates which plan is better
        idx = np.argmax(c != 0)
        return c[idx]


def mmphi(X: np.ndarray, q: Optional[float] = 2.0, p: Optional[float] = 1.0, verbosity=0) -> float:
    """
    Calculates the Morris-Mitchell sampling plan quality criterion.

    Args:
        X (np.ndarray):
            A 2D array representing the sampling plan, where each row is a point in
            d-dimensional space (shape: (n, d)).
        q (float, optional):
            Exponent used in the computation of the metric. Defaults to 2.0.
        p (float, optional):
            The distance norm to use. For example, p=1 is Manhattan (L1),
            p=2 is Euclidean (L2). Defaults to 1.0.
        verbosity (int, optional):
            If set to 1, prints additional information about the computation.
            Defaults to 0 (no additional output).

    Returns:
        float:
            The space-fillingness metric Phiq. Larger values typically indicate a more
            space-filling plan according to the Morris-Mitchell criterion.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.sampling import mmphi
        >>> # Simple 3-point sampling plan in 2D
        >>> X = np.array([
        ...     [0.0, 0.0],
        ...     [0.5, 0.5],
        ...     [1.0, 1.0]
        ... ])
        >>> # Calculate the space-fillingness metric with q=2, using Euclidean distances (p=2)
        >>> quality = mmphi(X, q=2, p=2)
        >>> print(quality)
        # This value indicates how well points are spread out, with smaller being better.
    """
    # check that X has unique rows
    if X.shape[0] != len(np.unique(X, axis=0)):
        # issue a warning if there are duplicate rows
        print("Warning: X contains duplicate rows. This may affect the space-fillingness metric.")
        # make X unique
        X = np.unique(X, axis=0)
    # Compute the distance multiplicities: J, and unique distances: d
    J, d = jd(X, p)
    print(f"J: {J}, d: {d}") if verbosity > 0 else None

    # Summation of J[i] * d[i]^(-q), then raised to 1/q
    # This follows the Morris-Mitchell definition.
    Phiq = np.sum(J * (d ** (-q))) ** (1.0 / q)
    return Phiq


def mmsort(X3D: np.ndarray, p: Optional[float] = 1.0) -> np.ndarray:
    """
    Ranks multiple sampling plans stored in a 3D array according to the
    Morris-Mitchell criterion, using a simple bubble sort.

    Args:
        X3D (np.ndarray):
            A 3D NumPy array of shape (n, d, m), where m is the number of
            sampling plans, and each plan is an (n, d) matrix of points.
        p (float, optional):
            The distance metric to use. p=1 for Manhattan (L1), p=2 for
            Euclidean (L2). Defaults to 1.0.

    Returns:
        np.ndarray:
            A 1D integer array of length m that holds the plan indices in
            ascending order of space-filling quality. The first index in the
            returned array corresponds to the most space-filling plan.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.sampling import mmsort
        >>> # Suppose we have two 3-point sampling plans in 2D, stored in X3D:
        >>> X1 = np.array([[0.0, 0.0],
        ...                [0.5, 0.5],
        ...                [1.0, 1.0]])
        >>> X2 = np.array([[0.2, 0.2],
        ...                [0.6, 0.4],
        ...                [0.9, 0.9]])
        >>> # Stack them along the third dimension: shape will be (3, 2, 2)
        >>> X3D = np.stack([X1, X2], axis=2)
        >>> # Sort them using the Morris-Mitchell criterion with p=2
        >>> ranking = mmsort(X3D, p=2.0)
        >>> print(ranking)
        # It might print [2 1] or [1 2], depending on which plan is more space-filling.
    """
    # Number of plans (m)
    m = X3D.shape[2]

    # Create index array (1-based to match original MATLAB convention)
    Index = np.arange(1, m + 1)

    swap_flag = True
    while swap_flag:
        swap_flag = False
        i = 0
        while i < m - 1:
            # Compare plan at Index[i] vs. Index[i+1] using mm()
            # Note: subtract 1 from each index to convert to 0-based array indexing
            if mm(X3D[:, :, Index[i] - 1], X3D[:, :, Index[i + 1] - 1], p) == 2:
                # Swap indices if the second plan is more space-filling
                Index[i], Index[i + 1] = Index[i + 1], Index[i]
                swap_flag = True
            i += 1

    return Index


def perturb(X: np.ndarray, PertNum: Optional[int] = 1) -> np.ndarray:
    """
    Performs a specified number of random element swaps on a sampling plan.
    If the plan is a Latin hypercube, the result remains a valid Latin hypercube.

    Args:
        X (np.ndarray):
            A 2D array (sampling plan) of shape (n, k), where each row is a point
            and each column is a dimension.
        PertNum (int, optional):
            The number of element swaps (perturbations) to perform. Defaults to 1.

    Returns:
        np.ndarray:
            The perturbed sampling plan, identical in shape to the input, with
            one or more random column swaps executed.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.sampling import perturb
        >>> # Create a simple 4x2 sampling plan
        >>> X_original = np.array([
        ...     [1, 3],
        ...     [2, 4],
        ...     [3, 1],
        ...     [4, 2]
        ... ])
        >>> # Perturb it once
        >>> X_perturbed = perturb(X_original, PertNum=1)
        >>> print(X_perturbed)
        # The output may differ due to random swaps, but each column is still a permutation of [1,2,3,4].
            [[1 3]
            [2 2]
            [3 1]
            [4 4]]
    """
    # Get dimensions of the plan
    n, k = X.shape
    if n < 2 or k < 2:
        raise ValueError("Latin hypercubes require at least 2 points and 2 dimensions")

    for _ in range(PertNum):
        # Pick a random column
        col = int(np.floor(np.random.rand() * k))

        # Pick two distinct row indices
        el1, el2 = 0, 0
        while el1 == el2:
            el1 = int(np.floor(np.random.rand() * n))
            el2 = int(np.floor(np.random.rand() * n))

        # Swap the two selected elements in the chosen column
        X[el1, col], X[el2, col] = X[el2, col], X[el1, col]

    return X


def mmlhs(X_start: np.ndarray, population: int, iterations: int, q: Optional[float] = 2.0, plot=False) -> np.ndarray:
    """
    Performs an evolutionary search (using perturbations) to find a Morris-Mitchell
    optimal Latin hypercube, starting from an initial plan X_start.

    This function does the following:
      1. Initializes a "best" Latin hypercube (X_best) from the provided X_start.
      2. Iteratively perturbs X_best to create offspring.
      3. Evaluates the space-fillingness of each offspring via the Morris-Mitchell
         metric (using mmphi).
      4. Updates the best plan whenever a better offspring is found.

    Args:
        X_start (np.ndarray):
            A 2D array of shape (n, k) providing the initial Latin hypercube
            (n points in k dimensions).
        population (int):
            Number of offspring to create in each generation.
        iterations (int):
            Total number of generations to run the evolutionary search.
        q (float, optional):
            The exponent used by the Morris-Mitchell space-filling criterion.
            Defaults to 2.0.
        plot (bool, optional):
            If True, a simple scatter plot of the first two dimensions will be
            displayed at each iteration. Only if k >= 2. Defaults to False.

    Returns:
        np.ndarray:
            A 2D array representing the most space-filling Latin hypercube found
            after all iterations, of the same shape as X_start.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.sampling import mmlhs
        >>> # Suppose we have an initial 4x2 plan
        >>> X_start = np.array([
        ...     [0, 0],
        ...     [1, 3],
        ...     [2, 1],
        ...     [3, 2]
        ... ])
        >>> # Search for a more space-filling plan
        >>> X_opt = mmlhs(X_start, population=5, iterations=10, q=2)
        >>> print("Optimized plan:")
        >>> print(X_opt)
    """
    n = X_start.shape[0]
    if n < 2:
        raise ValueError("Latin hypercubes require at least 2 points")
    k = X_start.shape[1]
    if k < 2:
        raise ValueError("Latin hypercubes are not defined for dim k < 2")

    # Initialize best plan and its metric
    X_best = X_start.copy()
    Phi_best = mmphi(X_best, q=q)

    # After 85% of iterations, reduce the mutation rate to 1
    leveloff = int(np.floor(0.85 * iterations))

    for it in range(1, iterations + 1):
        # Decrease number of mutations over time
        if it < leveloff:
            mutations = int(round(1 + (0.5 * n - 1) * (leveloff - it) / (leveloff - 1)))
        else:
            mutations = 1

        X_improved = X_best.copy()
        Phi_improved = Phi_best

        # Create offspring, evaluate, and keep the best
        for _ in range(population):
            X_try = perturb(X_best.copy(), mutations)
            Phi_try = mmphi(X_try, q=q)

            if Phi_try < Phi_improved:
                X_improved = X_try
                Phi_improved = Phi_try

        # Update the global best if we found a better plan
        if Phi_improved < Phi_best:
            X_best = X_improved
            Phi_best = Phi_improved

        # Simple visualization of the first two dimensions
        if plot and (X_best.shape[1] >= 2):
            plt.clf()
            plt.scatter(X_best[:, 0], X_best[:, 1], marker="o")
            plt.grid(True)
            plt.title(f"Iteration {it} - Current Best Plan")
            plt.pause(0.01)

    return X_best


def bestlh(n: int, k: int, population: int, iterations: int, p=1, plot=False, verbosity=0, edges=0, q_list=[1, 2, 5, 10, 20, 50, 100]) -> np.ndarray:
    """
    Generates an optimized Latin hypercube by evolving the Morris-Mitchell
    criterion across multiple exponents (q values) and selecting the best plan.

    Args:
        n (int):
            Number of points required in the Latin hypercube.
        k (int):
            Number of design variables (dimensions).
        population (int):
            Number of offspring in each generation of the evolutionary search.
        iterations (int):
            Number of generations for the evolutionary search.
        p (int, optional):
            The distance norm to use. p=1 for Manhattan (L1), p=2 for Euclidean (L2).
            Defaults to 1 (faster than 2).
        plot (bool, optional):
            If True, a scatter plot of the optimized plan in the first two dimensions
            will be displayed. Only if k>=2.  Defaults to False.
        verbosity (int, optional):
            Verbosity level. 0 is silent, 1 prints the best q value found. Defaults to 0.
        edges (int, optional):
            If 1, places centers of the extreme bins at the domain edges ([0,1]).
            Otherwise, bins are fully contained within the domain, i.e. midpoints.
            Defaults to 0.
        q_list (list, optional):
            A list of q values to optimize. Defaults to [1, 2, 5, 10, 20, 50, 100].
            These values are used to evaluate the space-fillingness of the Latin
            hypercube. The best plan is selected based on the lowest mmphi value.

    Returns:
        np.ndarray:
            A 2D array of shape (n, k) representing an optimized Latin hypercube.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> import numpy as np
            from spotpython.utils.sampling import bestlh
            bestlh(n=5, k=2, population=5, iterations=10)
    """
    if n < 2:
        raise ValueError("Latin hypercubes require at least 2 points")
    if k < 2:
        raise ValueError("Latin hypercubes are not defined for dim k < 2")

    # A list of exponents (q) to optimize

    # Start with a random Latin hypercube
    X_start = rlh(n, k, edges=edges)

    # Allocate a 3D array to store the results for each q
    # (shape: (n, k, number_of_q_values))
    X3D = np.zeros((n, k, len(q_list)))

    # Evolve the plan for each q in q_list
    for i, q_val in enumerate(q_list):
        if verbosity > 0:
            print(f"Now optimizing for q={q_val}...")
        X3D[:, :, i] = mmlhs(X_start, population, iterations, q_val)

    # Sort the set of evolved plans according to the Morris-Mitchell criterion
    index_order = mmsort(X3D, p=p)

    # index_order is a 1-based array of plan indices; the first element is the best
    best_idx = index_order[0] - 1
    if verbosity > 0:
        print(f"Best lh found using q={q_list[best_idx]}...")

    # The best plan in 3D array order
    X = X3D[:, :, best_idx]

    # Plot the first two dimensions
    if plot and (k >= 2):
        plt.scatter(X[:, 0], X[:, 1], c="r", marker="o")
        plt.title(f"Morris-Mitchell optimum plan found using q={q_list[best_idx]}")
        plt.xlabel("x_1")
        plt.ylabel("x_2")
        plt.grid(True)
        plt.show()

    return X


def phisort(X3D: np.ndarray, q: Optional[float] = 2.0, p: Optional[float] = 1.0) -> np.ndarray:
    """
    Ranks multiple sampling plans stored in a 3D array by the Morris-Mitchell
    numerical quality metric (mmphi). Uses a simple bubble-sort:
    sampling plans with smaller mmphi values are placed first in the index array.

    Args:
        X3D (np.ndarray):
            A 3D array of shape (n, d, m), where m is the number of sampling plans.
        q (float, optional):
            Exponent for the mmphi metric. Defaults to 2.0.
        p (float, optional):
            Distance norm for mmphi. p=1 is Manhattan; p=2 is Euclidean. Defaults to 1.0.

    Returns:
        np.ndarray:
            A 1D integer array of length m, giving the plan indices in ascending
            order of mmphi. The first index in the returned array corresponds
            to the numerically lowest mmphi value.

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> import numpy as np
            from spotpython.utils.sampling import phisort
            X1 = bestlh(n=5, k=2, population=5, iterations=10)
            X2 = bestlh(n=5, k=2, population=15, iterations=20)
            X3 = bestlh(n=5, k=2, population=25, iterations=30)
            # Map X1 and X2 so that X3D has the two sampling plans in X3D[:, :, 0] and X3D[:, :, 1]
            X3D = np.array([X1, X2])
            print(phisort(X3D))
            X3D = np.array([X3, X2])
            print(phisort(X3D))
                [2 1]
                [1 2]
    """
    # Number of 2D sampling plans
    m = X3D.shape[2]

    # Create a 1-based index array
    Index = np.arange(1, m + 1)

    # Bubble-sort: plan with lower mmphi() climbs toward the front
    swap_flag = True
    while swap_flag:
        swap_flag = False
        for i in range(m - 1):
            # Retrieve mmphi values for consecutive plans
            val_i = mmphi(X3D[:, :, Index[i] - 1], q=q, p=p)
            val_j = mmphi(X3D[:, :, Index[i + 1] - 1], q=q, p=p)

            # Swap if the left plan's mmphi is larger (i.e. 'worse')
            if val_i > val_j:
                Index[i], Index[i + 1] = Index[i + 1], Index[i]
                swap_flag = True

    return Index


def subset(X: np.ndarray, ns: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a space-filling subset of a given size from a sampling plan, along with
    the remainder. It repeatedly attempts to substitute each point in the subset
    with a point from the remainder if doing so improves the Morris-Mitchell metric.

    Args:
        X (np.ndarray):
            A 2D array representing the original sampling plan, of shape (n, d).
        ns (int):
            The size of the desired subset.

    Returns:
        (np.ndarray, np.ndarray):
            A tuple (Xs, Xr) where:
            - Xs is the chosen subset of size ns, with space-filling properties.
            - Xr is the remainder (X \\ Xs).

    Notes:
        Many thanks to the original author of this code, A Sobester, for providing the original Matlab code under the GNU Licence. Original Matlab Code: Copyright 2007 A Sobester:
        "This program is free software: you can redistribute it and/or modify  it
        under the terms of the GNU Lesser General Public License as published by
        the Free Software Foundation, either version 3 of the License, or any
        later version.
        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
        General Public License for more details.
        You should have received a copy of the GNU General Public License and GNU
        Lesser General Public License along with this program. If not, see
        <http://www.gnu.org/licenses/>."

    Examples:
        >>> from spotpython.utils.sampling import subset, bestlh
            X = bestlh(n=5, k=3, population=5, iterations=10)
            Xs, Xr = subset(X, ns=2)
            print(Xs)
            print(Xr)
                [[0.25 0.   0.5 ]
                [0.5  0.75 0.  ]]
                [[1.   0.25 0.25]
                [0.   1.   0.75]
                [0.75 0.5  1.  ]]
    """
    # Number of total points
    n = X.shape[0]

    # Morris-Mitchell parameters
    p = 1
    q = 5

    # Create a random permutation of row indices
    r = np.random.permutation(n)

    # Initial subset and remainder
    Xs = X[r[:ns], :].copy()
    Xr = X[r[ns:], :].copy()

    # Attempt to improve space-filling by swapping points
    for j in range(ns):
        orig_crit = mmphi(Xs, q=q, p=p)
        orig_point = Xs[j, :].copy()

        # Track best substitution index and metric
        bestsub = 0
        bestsubcrit = np.inf

        # Try replacing Xs[j] with each candidate in Xr
        for i in range(n - ns):
            Xs[j, :] = Xr[i, :]
            crit = mmphi(Xs, q=q, p=p)
            if crit < bestsubcrit:
                bestsubcrit = crit
                bestsub = i

        # If a better subset is found, swap permanently
        if bestsubcrit < orig_crit:
            Xs[j, :] = Xr[bestsub, :].copy()
            Xr[bestsub, :] = orig_point
        else:
            Xs[j, :] = orig_point

    return Xs, Xr


def mmphi_intensive(X: np.ndarray, q: Optional[float] = 2.0, p: Optional[float] = 2.0) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculates a size-invariant Morris-Mitchell criterion.

    This "intensive" version of the criterion allows for the comparison of
    sampling plans with different sample sizes by normalizing for the number
    of point pairs. A smaller value indicates a better (more space-filling)
    design.

    Args:
        X (np.ndarray):
            A 2D array representing the sampling plan (shape: (n, d)).
        q (float, optional):
            The exponent used in the computation of the metric. Defaults to 2.0.
        p (float, optional):
            The distance norm to use (e.g., p=1 for Manhattan, p=2 for Euclidean).
            Defaults to 2.0.

    Returns:
        tuple[float, np.ndarray, np.ndarray]:
            A tuple containing:
            - intensive_phiq: The intensive space-fillingness metric.
            - J: Multiplicities of distances.
            - d: Unique distances.

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.sampling import mmphi_intensive
        >>> # Create a simple 3-point sampling plan in 2D
        >>> X = np.array([
        ...     [0.0, 0.0],
        ...     [0.5, 0.5],
        ...     [1.0, 1.0]
        ... ])
        >>> # Calculate the intensive space-fillingness metric with q=2, using Euclidean distances (p=2)
        >>> quality, J, d = mmphi_intensive(X, q=2, p=2)
        >>> print(quality)
    """
    # Ensure there are no duplicate points
    if X.shape[0] != len(np.unique(X, axis=0)):
        X = np.unique(X, axis=0)

    n_points = X.shape[0]

    # The criterion is not well-defined for fewer than 2 points.
    if n_points < 2:
        return np.inf, 0, 0

    # Get the unique distances and their multiplicities
    J, d = jd(X, p=p)

    # If all points are identical, the design is infinitely bad.
    if d.size == 0:
        return np.inf, J, d

    # Calculate the number of unique pairs of points
    M = n_points * (n_points - 1) / 2

    try:
        # Calculate the sum term of the original mmphi
        sum_term = np.sum(J * (d ** (-q)))
        # Normalize the sum by M before taking the final root
        intensive_phiq = (sum_term / M) ** (1.0 / q)
    except ZeroDivisionError:
        return np.inf
    except FloatingPointError:
        return np.inf
    except Exception:
        return np.inf

    return intensive_phiq, J, d


def mmphi_intensive_update(X: np.ndarray, new_point: np.ndarray, J: np.ndarray, d: np.ndarray, q: float = 2.0, p: float = 2.0) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Updates the Morris-Mitchell intensive criterion for n+1 points by adding a new point to the design.

    Args:
        X (np.ndarray): Existing sampling plan (shape: (n, d)).
        new_point (np.ndarray): New point to add (shape: (d,)).
        J (np.ndarray): Multiplicities of distances for the existing design.
        d (np.ndarray): Unique distances for the existing design.
        q (float): Exponent used in the computation of the metric. Defaults to 2.0.
        p (float): Distance norm to use (e.g., p=1 for Manhattan, p=2 for Euclidean). Defaults to 2.0.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: Updated intensive_phiq, updated_J, updated_d.

    Examples:
        >>> import numpy as np
        >>> from spotpython.utils.sampling import mmphi_intensive_update
        >>> # Existing design with 3 points in 2D
        >>> X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        >>> phiq, J, d = mmphi_intensive(X, q=2, p=2)
        >>> # New point to add
        >>> new_point = np.array([0.1, 0.1])
        >>> # Update the intensive criterion
        >>> updated_phiq, updated_J, updated_d = mmphi_intensive_update(X, new_point, J, d, q=2, p=2)

    """
    n_points = X.shape[0]
    if n_points < 1:
        raise ValueError("The existing design must contain at least one point.")

    # Compute distances between the new point and all existing points
    new_distances = np.array([np.linalg.norm(new_point - X[i], ord=p) for i in range(n_points)])

    # Combine old distances and new distances into a single list
    all_distances = []
    for dist, count in zip(d, J):
        all_distances.extend([dist] * count)
    all_distances.extend(new_distances)

    # Find unique distances and their counts
    updated_d, updated_J = np.unique(all_distances, return_counts=True)

    # Calculate the number of unique pairs of points
    M = (n_points + 1) * n_points / 2

    # Compute the updated intensive_phiq
    sum_term = np.sum(updated_J * (updated_d ** (-q)))
    intensive_phiq = (sum_term / M) ** (1.0 / q)

    return intensive_phiq, updated_J, updated_d
