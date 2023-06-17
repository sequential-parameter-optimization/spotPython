# Purpose: Functions for the VBDP project


def cluster_features(X):
    """Clusters the features of a dataframe based on similarity

    Args:
        X (pd.DataFrame): dataframe with features
    Returns:
        X (pd.DataFrame): dataframe with new features
    Examples:
        >>> df = pd.DataFrame({"a": [True, False, True], "b": [True, True, False], "c": [False, False, True]})
        >>> df
            a      b      c
        0  True   True  False
        1 False   True  False
        2  True  False   True
        >>> cluster_features(df)
            a      b      c  cluster
        0  True   True  False       0
        1 False   True  False       1
        2  True  False   True        2
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


def affinity_propagation_features(X):
    """Clusters the features of a dataframe using Affinity Propagation

    Args:
        X (pd.DataFrame): dataframe with features
    Returns:
        X (pd.DataFrame): dataframe with new features
    Examples:
        >>> df = pd.DataFrame({"a": [True, False, True], "b": [True, True, False], "c": [False, False, True]})
        >>> df
            a      b      c
        0  True   True   False
        1  False  True   False
        2  True   False  True
        >>> affinity_propagation_features(df)
            a      b      c  cluster
        0  True   True   False       0
        1  False  True   False       1
        2  True   False  True        2
    """
    from sklearn.cluster import AffinityPropagation
    from sklearn.metrics.pairwise import manhattan_distances

    D = manhattan_distances(X)
    af = AffinityPropagation(random_state=0, affinity="precomputed").fit(D)
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    print("Estimated number of clusters: %d" % n_clusters_)
    X["cluster"] = af.labels_
    return X
