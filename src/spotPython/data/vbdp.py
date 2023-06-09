import itertools
import pandas as pd


def cluster_features(df_vbdp):
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
