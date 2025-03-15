import numpy as np
from typing import Tuple, Optional


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

    Example:
        >>> q = [3, 4]
        >>> X = fullfactorial(q, Edges=1)
        >>> print(X)
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

    Example:
        >>> import numpy as np
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

    Example:
        >>> import numpy as np
        >>> from your_module import jd
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

    Example:
        >>> import numpy as np
        >>> from your_module import mm
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


def mmphi(X: np.ndarray, q: Optional[float] = 2.0, p: Optional[float] = 1.0) -> float:
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

    Returns:
        float:
            The space-fillingness metric Phiq. Larger values typically indicate a more
            space-filling plan according to the Morris-Mitchell criterion.

    Example:
        >>> import numpy as np
        >>> from your_module import mmphi
        >>> # Simple 3-point sampling plan in 2D
        >>> X = np.array([
        ...     [0.0, 0.0],
        ...     [0.5, 0.5],
        ...     [1.0, 1.0]
        ... ])
        >>> # Calculate the space-fillingness metric with q=2, using Euclidean distances (p=2)
        >>> quality = mmphi(X, q=2, p=2)
        >>> print(quality)
        # This value indicates how well points are spread out, with higher being better.
    """
    # Compute the distance multiplicities: J, and unique distances: d
    J, d = jd(X, p)

    # Summation of J[i] * d[i]^(-q), then raised to 1/q
    # This follows the Morris-Mitchell definition.
    Phiq = np.sum(J * (d ** (-q))) ** (1.0 / q)
    return Phiq
