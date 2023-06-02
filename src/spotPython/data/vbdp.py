import itertools
import pandas as pd


def modify_vbdp_dataframe(df_vbdp):
    """Modify the dataframe with extended features for the vbdp model.
    Args:
        df_vbdp (pandas.DataFrame): The vbdp dataframe to modify.
    Returns:
        pandas.DataFrame: The modified dataframe.
    """
    # Determine the n most common diseases for each prognosis
    for n in range(3, 6):
        # Group the data by 'prognosis' column and count the occurrence of each disease
        disease_counts = df_vbdp.iloc[:, :64].groupby(df_vbdp["prognosis"]).sum()
        # Get the top n diseases for each prognosis
        top_diseases = disease_counts.apply(lambda x: x.nlargest(n).index.tolist(), axis=1)
        # Create new columns for each prognosis-disease combination
        for prognosis, diseases in top_diseases.items():
            for disease in diseases:
                column_name = f"{prognosis}_{disease}_{n}"
                df_vbdp[column_name] = 0
        # Iterate through the dataframe rows and update the new columns
        for index, row in df_vbdp.iterrows():
            prognosis = row["prognosis"]
            diseases = top_diseases[prognosis]
            for disease in diseases:
                column_name = f"{prognosis}_{disease}_{n}"
                if row[disease] == 1:
                    df_vbdp.at[index, column_name] = 1
    # Target should be the last column
    df_vbdp = df_vbdp[[c for c in df_vbdp if c not in ["prognosis"]] + ["prognosis"]]
    return df_vbdp


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
