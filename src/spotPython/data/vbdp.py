# Purpose: Functions for the VBDP project


import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import manhattan_distances


def cluster_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clusters the features of a dataframe based on similarity.

    This function takes a dataframe with features and clusters them based on similarity.
    The resulting dataframe contains the original features as well as new features representing the clusters.

    Args:
        X (pd.DataFrame): A dataframe with features.

    Returns:
        (pd.DataFrame): A dataframe with the original features and new cluster features.

    Examples:
        >>> df = pd.DataFrame({"a": [True, False, True], "b": [True, True, False], "c": [False, False, True]})
        >>> df
            a      b      c
        0  True   True  False
        1 False   True  False
        2  True  False   True
        >>> cluster_features(df)
            a      b      c  c_0  c_1  c_2  c_3
        0  True   True  False    0    0    0    0
        1 False   True  False    0    0    0    0
        2  True  False   True    0    0    0    0
    """
    c_0 = X.columns[X.columns.str.contains("pain")]
    c_1 = X.columns[X.columns.str.contains("inflammation")]
    c_2 = X.columns[X.columns.str.contains("bleed")]
    c_3 = X.columns[X.columns.str.contains("skin")]
    X["c_0"] = X[c_0].sum(axis=1)
    X["c_1"] = X[c_1].sum(axis=1)
    X["c_2"] = X[c_2].sum(axis=1)
    X["c_3"] = X[c_3].sum(axis=1)
    return X


def affinity_propagation_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clusters the features of a dataframe using Affinity Propagation.

    This function takes a dataframe with features and clusters them using the
    Affinity Propagation algorithm. The resulting dataframe contains the original
    features as well as a new feature representing the cluster labels.

    Args:
        X (pd.DataFrame):
            A dataframe with features.

    Returns:
        (pd.DataFrame):
            A dataframe with the original features and a new cluster feature.

    Examples:
        >>> df = pd.DataFrame({"a": [True, False, True], "b": [True, True, False], "c": [False, False, True]})
        >>> df
            a      b      c
        0  True   True   False
        1  False  True   False
        2  True   False  True
        >>> affinity_propagation_features(df)
        Estimated number of clusters: 3
            a      b      c  cluster
        0  True   True   False       0
        1  False  True   False       1
        2  True   False  True        2
    """
    D = manhattan_distances(X)
    af = AffinityPropagation(random_state=0, affinity="precomputed").fit(D)
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    print("Estimated number of clusters: %d" % n_clusters_)
    X["cluster"] = af.labels_
    return X
