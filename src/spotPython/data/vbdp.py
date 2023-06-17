import itertools
import pandas as pd


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


def combine_features(X):
    """Combines all features in a dataframe with each other using bitwise operations

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
            >>> combine_features(df)
                a      b      c  a_and_b  a_or_b  a_xor_b  a_and_c  a_or_c  a_xor_c  b_and_c  b_or_c  b_xor_c
            0  True   True  False     True    True    False    False    True     True    False    True     True
            1 False   True  False    False    True     True    False   False    False    False   False    False
            2  True  False   True    False    True     True     True    True    False    False    True     True
    """
    new_cols = []
    # Iterate over all pairs of columns
    for col1, col2 in itertools.combinations(X.columns, 2):
        # Create new columns for the bitwise AND, OR and XOR operations
        and_col = X[[col1, col2]].apply(lambda x: x[col1] & x[col2], axis=1)
        or_col = X[[col1, col2]].apply(lambda x: x[col1] | x[col2], axis=1)
        xor_col = X[[col1, col2]].apply(lambda x: x[col1] ^ x[col2], axis=1)
        new_cols.extend([and_col, or_col, xor_col])
    # Join all the new columns at once
    X = pd.concat([X] + new_cols, axis=1)
    return X


def symptom_features(X, y):
    """Generate new features based on the joint symptoms of a disease
    Args:
        X (pd.DataFrame): dataframe with features
        y (pd.Series): series with target values
    """
    # Combine X and y into one dataframe
    Xy = pd.concat([X, y], axis=1)
    # Add names to the columns: x1, x2, ..., xn, y
    Xy.columns = ["x" + str(i) for i in range(1, X.shape[1] + 1)] + ["y"]
    # full train data with X and y values
    marginals = Xy.groupby("y").mean()
    top_2_symptoms = {}
    bot_2_symptoms = {}
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
        top_2_symptoms[name] = dic
        bot_2_symptoms[name] = dic_bot
        combinations.append(((top_1, top_2), (bot_1, bot_2)))
    Xy_mod = Xy.copy()
    convert = Xy.drop(columns=["y"]).columns.values
    for val in convert:
        Xy_mod[val] = Xy_mod[val].astype("int")
    for group in combinations:
        for comb in group:
            col1, col2 = comb
            new_columns = pd.DataFrame(
                {
                    f"{col1}_and_{col2}": Xy_mod[col1] & Xy_mod[col2],
                    f"{col1}_or_{col2}": Xy_mod[col1] | Xy_mod[col2],
                    f"{col1}_xor_{col2}": Xy_mod[col1] ^ Xy_mod[col2],
                }
            )
            Xy_mod = pd.concat([Xy_mod, new_columns], axis=1)
    # removing duplicate features
    Xy_mod = Xy_mod.loc[:, ~Xy_mod.columns.duplicated()].copy()
    print(f"Number of features: {Xy_mod.shape[1]}")
    print(f"Number of samples: {Xy_mod.shape[0]}")
    # remove the column y from the Xy_mod data frame
    X_mod = Xy_mod.drop(columns=["y"])
    # print the column names
    print(f"Column names: {Xy_mod.columns.values}")
    # X_new = add_logical_columns(X_mod, 2)
    X_new = combine_features(X_mod)
    print(f"Number of features: {X_new.shape[1]}")
    print(f"Number of samples: {X_new.shape[0]}")
    return X_new, top_2_symptoms, bot_2_symptoms
