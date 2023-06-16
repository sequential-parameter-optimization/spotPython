import itertools
import pandas as pd


def cluster_features(df_vbdp):
    """Clusters the features of a dataframe based on similarity

    Args:
        df_vbdp (pd.DataFrame): dataframe with features
    Returns:
        df_vbdp (pd.DataFrame): dataframe with new features
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
    c_0 = df_vbdp.columns[df_vbdp.columns.str.contains("pain")]
    c_1 = df_vbdp.columns[df_vbdp.columns.str.contains("inflammation")]
    c_2 = df_vbdp.columns[df_vbdp.columns.str.contains("bleed")]
    c_3 = df_vbdp.columns[df_vbdp.columns.str.contains("skin")]
    df_vbdp["c_0"] = df_vbdp[c_0].sum(axis=1)
    df_vbdp["c_1"] = df_vbdp[c_1].sum(axis=1)
    df_vbdp["c_2"] = df_vbdp[c_2].sum(axis=1)
    df_vbdp["c_3"] = df_vbdp[c_3].sum(axis=1)
    return df_vbdp


def affinity_propagation_features(df_vbdp):
    """Clusters the features of a dataframe using Affinity Propagation

    Args:
        df_vbdp (pd.DataFrame): dataframe with features
    Returns:
        df_vbdp (pd.DataFrame): dataframe with new features
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

    X = manhattan_distances(df_vbdp)
    af = AffinityPropagation(random_state=0, affinity="precomputed").fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    print("Estimated number of clusters: %d" % n_clusters_)
    df_vbdp["cluster"] = af.labels_
    return df_vbdp


def combine_features(df_vbdp):
    """Combines all features in a dataframe with each other using bitwise operations

    Args:
        df_vbdp (pd.DataFrame): dataframe with features
    Returns:
        df_vbdp (pd.DataFrame): dataframe with new features
        Examples:
            >>> df = pd.DataFrame({"a": [True, False, True], "b": [True, True, False], "c": [False, False, True]})
            >>> df
                a      b      c
            0  True   True  False
            1 False   True  False
            2  True  False   True
            >>> combine_features(df)
                a      b      c  a_and_b  a_or_b  a_xor_b  a_and_c  a_or_c  a_xor_c  b_and_c  b_or_c  b_xor_c
            0  True   True  False     True    True    False    False    True     True    False    True     True
            1 False   True  False    False    True     True    False   False    False    False   False    False
            2  True  False   True    False    True     True     True    True    False    False    True     True
    """
    new_cols = []
    # Iterate over all pairs of columns
    for col1, col2 in itertools.combinations(df_vbdp.columns, 2):
        # Create new columns for the bitwise AND, OR and XOR operations
        and_col = df_vbdp[[col1, col2]].apply(lambda x: x[col1] & x[col2], axis=1)
        or_col = df_vbdp[[col1, col2]].apply(lambda x: x[col1] | x[col2], axis=1)
        xor_col = df_vbdp[[col1, col2]].apply(lambda x: x[col1] ^ x[col2], axis=1)
        new_cols.extend([and_col, or_col, xor_col])
    # Join all the new columns at once
    df_vbdp = pd.concat([df_vbdp] + new_cols, axis=1)
    return df_vbdp
