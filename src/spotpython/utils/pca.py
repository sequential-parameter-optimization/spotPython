import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def pca_analysis(
    df,
    df_name="",
    k=10,
    scaler=StandardScaler(),
    max_scree=None,
    figsize=(12, 6),
) -> tuple:
    """
    Perform PCA analysis on a DataFrame with specified scaling.

    Args:
        df (pd.DataFrame):
            The input data frame to perform PCA on.
        df_name (str):
            The name of the data frame.
        k (int):
            The number of top features to select based on their influence on PC1.
        scaler (obj):
            An instance of a Scaler from sklearn (e.g., StandardScaler()).
        max_scree (int):
            The maximum number of principal components to plot in the scree plot. Default is None, which means all components will be plotted.
        figsize (tuple):
            The size of the figure for the plots (width, height).

    Returns:
        tuple: Two pd.Index objects containing the names of the top k features most influential on PC1 and PC2, respectively.

    Examples:
        >>> import pandas as pd
        >>> from spotpython.utils import pca_analysis
        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [1, 2, 3],
        ...     "C": [4, 5, 6]
        ... })
        >>> pca_analysis(df)
    """
    # Scale the data
    scaled_data = scaler.fit_transform(df)
    feature_names = df.columns
    sample_names = df.index

    # Perform PCA
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    # Scree plot
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    full_labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]

    # Limit the number of PCs in the scree plot
    if max_scree is not None:
        per_var = per_var[:max_scree]
        scree_labels = full_labels[:max_scree]
    else:
        scree_labels = full_labels

    plt.figure(figsize=figsize)  # Set the figure size for the scree plot
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=scree_labels)
    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Component")
    plt.title(f"Scree Plot. {df_name}")
    plt.show()

    # PCA plot
    plt.figure(figsize=figsize)  # Set the figure size for the PCA plot
    pca_df = pd.DataFrame(pca_data, index=sample_names, columns=full_labels)

    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title(f"PCA Graph. {df_name}")
    plt.xlabel("PC1 - {0}%".format(per_var[0]))
    plt.ylabel("PC2 - {0}%".format(per_var[1]))

    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

    plt.show()

    # Determine top k features influencing PC1 and PC2
    loading_scores_pc1 = pd.Series(pca.components_[0], index=feature_names)
    loading_scores_pc2 = pd.Series(pca.components_[1], index=feature_names)

    sorted_loading_scores_pc1 = loading_scores_pc1.abs().sort_values(ascending=False)
    sorted_loading_scores_pc2 = loading_scores_pc2.abs().sort_values(ascending=False)

    top_k_features_pc1 = sorted_loading_scores_pc1.head(k).index
    top_k_features_pc2 = sorted_loading_scores_pc2.head(k).index

    return top_k_features_pc1, top_k_features_pc2
