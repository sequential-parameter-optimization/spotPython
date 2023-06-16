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


def js_combine(df, target_column="prognosis"):
    """Combines all features in a dataframe with each other using bitwise operations

    Args:
        df (pd.DataFrame): dataframe with features
        target_column (str): name of the target column
    Returns:
        df (pd.DataFrame): dataframe with new features
    Examples:
        >>> df = pd.DataFrame({"a": [True, False, True], "b": [True, True, False], "c": [False, False, True]})
        >>> df
            a      b      c
        0  True   True  False
        1 False   True  False
        2  True  False   True
        >>> js_combine(df)
            a      b      c  a_and_b  a_or_b  a_xor_b  a_and_c  a_or_c  a_xor_c  b_and_c  b_or_c  b_xor_c
        0  True   True  False     True    True    False    False    True     True    False    True     True
        1 False   True  False    False    True     True    False   False    False    False   False    False
        2  True  False   True    False    True     True     True    True    False    False    True     True
    """
    for col1, col2 in itertools.combinations(df.drop(columns=[target_column]).columns, 2):
        df[f"{col1}_and_{col2}"] = df[col1] & df[col2]
        df[f"{col1}_or_{col2}"] = df[col1] | df[col2]
        df[f"{col1}_xor_{col2}"] = df[col1] ^ df[col2]
    return df


def js_features(train, target_column="prognosis"):
    """Generates new features based on the joint symptoms of a disease

    Args:
        train (pd.DataFrame): dataframe with features
        target_column (str): name of the target column
    Returns:
        train_mod (pd.DataFrame): dataframe with new features
    Examples:
        >>> df = pd.DataFrame({"a": [True, False, True],
                               "b": [True, True, False],
                               "c": [False, False, True],
                               "prognosis": ["disease1",
                                             "disease2",
                                             "disease1"]})
        >>> df
            a      b      c prognosis
        0  True   True  False  disease1
        1 False   True  False  disease2
        2  True  False   True  disease1
        >>> js_features(df)
            a      b      c prognosis  a_and_b  a_or_b  a_xor_b  a_and_c  a_or_c  a_xor_c  b_and_c  b_or_c  b_xor_c
        0  True   True  False  disease1     True    True    False    False    True     True    False    True     True
        1 False   True  False  disease2    False    True     True    False   False    False    False   False    False
        2  True  False   True  disease1    False    True     True     True    True    False    False    True     True
    """
    # full train data with X and y values
    marginals = train.groupby(target_column).mean()
    top_2_symptopms = {}
    bot_2_symtopms = {}
    # for feature generation
    combinations = []
    for i in range(marginals.shape[0]):
        symptoms = marginals.iloc[i]
        # for b in True, False:
        sorted = symptoms.sort_values(ascending=False)
        top_1 = sorted.keys()[0]
        top_1_per = sorted.values[0]
        top_2 = sorted.keys()[1]
        top_2_per = sorted.values[1]

        bot_1 = sorted.keys()[-1]
        bot_1_per = sorted.values[-1]
        bot_2 = sorted.keys()[-2]
        bot_2_per = sorted.values[-2]

        name = marginals.index[i]
        dic = {top_1: top_1_per, top_2: top_2_per}
        dic_bot = {bot_1: bot_1_per, bot_2: bot_2_per}
        top_2_symptopms[name] = dic
        bot_2_symtopms[name] = dic_bot
        combinations.append(((top_1, top_2), (bot_1, bot_2)))
    train_mod = train.copy()
    convert = train.drop(columns=["prognosis"]).columns.values
    for val in convert:
        train_mod[val] = train_mod[val].astype("int")
    for group in combinations:
        for comb in group:
            col1, col2 = comb
            new_columns = pd.DataFrame(
                {
                    f"{col1}_and_{col2}": train_mod[col1] & train_mod[col2],
                    f"{col1}_or_{col2}": train_mod[col1] | train_mod[col2],
                    f"{col1}_xor_{col2}": train_mod[col1] ^ train_mod[col2],
                }
            )
            train_mod = pd.concat([train_mod, new_columns], axis=1)
    # removing duplicate features
    train_mod = train_mod.loc[:, ~train_mod.columns.duplicated()].copy()
    train_mod = js_combine(train_mod)
    return train_mod, top_2_symptopms, bot_2_symtopms
