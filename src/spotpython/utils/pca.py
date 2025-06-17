import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def get_pca(df, n_components=3) -> tuple:
    """
    Scale the numeric data and perform PCA.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_components (int):
            Number of principal components to compute.
            Defaults to 3.

    Returns:
        tuple:
            - pca (PCA): Fitted PCA object.
            - scaled_data (np.ndarray): Scaled numeric data.
            - feature_names (pd.Index): Names of the numeric features.
            - sample_names (pd.Index): Index of the samples.
            - pca_data (np.ndarray): PCA-transformed data.

    Examples:
        >>> import pandas as pd
        >>> from spotpython.utils.pca import get_pca
        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6],
        ...     "C": ["x", "y", "z"]  # Non-numeric column will be ignored
        ... })
        >>> pca, scaled_data, feature_names, sample_names, pca_data = get_pca(df)
        >>> print(feature_names)
        Index(['A', 'B'], dtype='object')
        >>> print(pca_data.shape)
        (3, 2)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    feature_names = numeric_df.columns
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(numeric_df)
    pca_columns = [f"PC{i+1}" for i in range(pca_scores.shape[1])]
    df_pca_components = pd.DataFrame(data=pca_scores, columns=pca_columns)
    sample_names = df.index
    return pca, pca_scores, feature_names, sample_names, df_pca_components


def plot_pca_scree(pca, df_name="", max_scree=None, figsize=(12, 6)) -> None:
    """Plot the scree plot for Principal Component Analysis (PCA).

    A scree plot shows the percentage of variance explained by each principal
    component in descending order. It helps in determining the optimal number
    of components to retain.

    Args:
        pca (sklearn.decomposition.PCA): Fitted PCA object containing the
            explained variance ratios.
        df_name (str, optional): Name of the dataset to be displayed in the plot title.
            Defaults to empty string.
        max_scree (int, optional): Maximum number of principal components to plot.
            If None, all components are plotted. Defaults to None.
        figsize (tuple, optional): Size of the figure as (width, height).
            Defaults to (12, 6).

    Returns:
        None: The function creates and displays a matplotlib plot.

    Examples:
        >>> import numpy as np
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.datasets import load_iris
        >>> from spotpython.utils.pca import plot_pca_scree
        >>>
        >>> # Load iris dataset
        >>> iris = load_iris()
        >>> X = iris.data
        >>>
        >>> # Fit PCA
        >>> pca = PCA()
        >>> pca.fit(X)
        >>>
        >>> # Create scree plot
        >>> plot_pca_scree(pca,
        ...                df_name="Iris Dataset",
        ...                max_scree=4,
        ...                figsize=(10, 5))
    """
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    full_labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]

    # Limit the number of PCs in the scree plot
    if max_scree is not None:
        per_var = per_var[:max_scree]
        scree_labels = full_labels[:max_scree]
    else:
        scree_labels = full_labels

    plt.figure(figsize=figsize)
    plt.plot(range(1, len(per_var) + 1), per_var, marker="o", linestyle="--")
    plt.xticks(range(1, len(per_var) + 1), scree_labels)
    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Component")
    plt.title(f"Scree Plot. {df_name}")
    plt.grid(True)
    plt.show()


def plot_pca1vs2(pca, pca_data, df_name="", figsize=(12, 6)) -> None:
    """Create a scatter plot of the first two principal components from PCA.

    This function visualizes the first two principal components (PC1 vs PC2) from a PCA analysis,
    creating a scatter plot where each point represents a sample in the transformed space.
    The percentage of variance explained by each component is shown on the axes.

    Args:
        pca (sklearn.decomposition.PCA): Fitted PCA object containing the explained
            variance ratios and components.
        pca_data (array-like): PCA-transformed data, where each row represents a sample
            and each column represents a principal component.
        df_name (str, optional): Name of the dataset to be displayed in the plot title.
            Defaults to empty string.
        figsize (tuple, optional): Size of the figure as (width, height).
            Defaults to (12, 6).

    Returns:
        None: The function creates and displays a matplotlib plot.

    Examples:
        >>> import numpy as np
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.datasets import load_iris
        >>> from spotpython.utils.pca import plot_pca1vs2
        >>>
        >>> # Load and prepare the iris dataset
        >>> iris = load_iris()
        >>> X = iris.data
        >>>
        >>> # Fit PCA and transform the data
        >>> pca = PCA()
        >>> pca_data = pca.fit_transform(X)
        >>>
        >>> # Create PCA scatter plot
        >>> plot_pca1vs2(pca,
        ...             pca_data,
        ...             df_name="Iris Dataset",
        ...             figsize=(10, 5))

    Note:
        - The function assumes that the input data has at least two principal components
        - Sample names are taken from the index of the created DataFrame
        - The percentage of variance explained is rounded to 1 decimal place
    """
    pca_df = pd.DataFrame(pca_data, columns=["PC" + str(i + 1) for i in range(pca_data.shape[1])])

    plt.figure(figsize=figsize)
    plt.scatter(pca_df["PC1"], pca_df["PC2"])
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    plt.title(f"PCA Graph. {df_name}")
    plt.xlabel(f"PC1 - {np.round(pca.explained_variance_ratio_[0] * 100, 1)}%")
    plt.ylabel(f"PC2 - {np.round(pca.explained_variance_ratio_[1] * 100, 1)}%")
    plt.grid(True)
    plt.show()


def get_pca_topk(pca, feature_names, k=10) -> tuple:
    """Identify the top k features that have the strongest influence on PC1 and PC2.

    This function analyzes the loading scores (coefficients) of the first two principal
    components to determine which original features contribute most strongly to these
    components. The absolute values of the loading scores are used to rank feature
    importance.

    Args:
        pca (sklearn.decomposition.PCA): Fitted PCA object containing the components_
            attribute with the principal components.
        feature_names (list-like): Names of the original features, must match the
            order of features used in PCA fitting.
        k (int, optional): Number of top features to select for each principal
            component. Defaults to 10.

    Returns:
        tuple: A tuple containing two lists:
            - list[str]: Names of the k features most influential on PC1
            - list[str]: Names of the k features most influential on PC2

    Examples:
        >>> import numpy as np
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.datasets import load_iris
        >>> from spotpython.utils.pca import get_pca_topk
        >>>
        >>> # Load and prepare the iris dataset
        >>> iris = load_iris()
        >>> X = iris.data
        >>> feature_names = iris.feature_names
        >>>
        >>> # Fit PCA
        >>> pca = PCA()
        >>> pca.fit(X)
        >>>
        >>> # Get top 2 most influential features for PC1 and PC2
        >>> top_pc1, top_pc2 = get_pca_topk(pca,
        ...                                 feature_names=feature_names,
        ...                                 k=2)
        >>> print("Top PC1 features:", top_pc1)
        >>> print("Top PC2 features:", top_pc2)

    Note:
        - The function assumes that PCA has been fitted on standardized data
        - The length of feature_names must match the number of features in the PCA input
        - k should not exceed the total number of features
    """
    loading_scores_pc1 = pd.Series(pca.components_[0], index=feature_names)
    loading_scores_pc2 = pd.Series(pca.components_[1], index=feature_names)

    sorted_loading_scores_pc1 = loading_scores_pc1.abs().sort_values(ascending=False)
    sorted_loading_scores_pc2 = loading_scores_pc2.abs().sort_values(ascending=False)

    top_k_features_pc1 = sorted_loading_scores_pc1.head(k).index.tolist()
    top_k_features_pc2 = sorted_loading_scores_pc2.head(k).index.tolist()

    return top_k_features_pc1, top_k_features_pc2


def get_loading_scores(pca, feature_names) -> pd.DataFrame:
    """Computes the loading scores matrix for Principal Component Analysis (PCA).

    Creates and returns a DataFrame showing how each original feature contributes
    to each principal component.

    Args:
        pca (sklearn.decomposition.PCA): Fitted PCA object containing the components_
            attribute with the principal components.
        feature_names (list-like): Names of the original features, must match the
            order of features used in PCA fitting.

    Returns:
        pd.DataFrame: DataFrame containing the loading scores matrix with features
            as rows and principal components as columns.

    Example:
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.datasets import load_iris
        >>> from spotpython.utils.pca import print_loading_scores,
        >>>
        >>> # Load and prepare iris dataset
        >>> iris = load_iris()
        >>> X = iris.data
        >>> feature_names = iris.feature_names
        >>>
        >>> # Fit PCA
        >>> pca = PCA()
        >>> pca.fit(X)
        >>>
        >>> # Print loading scores
        >>> scores_df = print_loading_scores(pca, feature_names)
        >>> print(scores_df)
    """
    loading_scores = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=feature_names)
    return loading_scores


def plot_loading_scores(loading_scores, figsize=(12, 8)) -> None:
    """Creates a heatmap visualization of PCA loading scores.

    Generates a heatmap showing the relationship between original features and
    principal components, with color intensity indicating the strength and
    direction of the relationship.

    Args:
        loading_scores (pd.DataFrame): DataFrame containing the loading scores
            matrix with features as rows and principal components as columns.
        figsize (tuple, optional): Size of the figure as (width, height).
            Defaults to (12, 8).

    Returns:
        None: The function creates and displays a matplotlib plot.

    Example:
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.datasets import load_iris
        >>> from spotpython.utils.pca import print_loading_scores, plot_loading_scores
        >>>
        >>> # Load and prepare iris dataset
        >>> iris = load_iris()
        >>> X = iris.data
        >>> feature_names = iris.feature_names
        >>>
        >>> # Fit PCA and get loading scores
        >>> pca = PCA()
        >>> pca.fit(X)
        >>> scores_df = print_loading_scores(pca, feature_names)
        >>>
        >>> # Create heatmap
        >>> plot_loading_scores(scores_df, figsize=(10, 6))
    """
    plt.figure(figsize=figsize)
    sns.heatmap(loading_scores, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Loading Score"}, linewidths=0.5)
    plt.title("PCA Loading Scores Heatmap")
    plt.xlabel("Principal Components")
    plt.ylabel("Original Features")
    plt.tight_layout()
    plt.show()


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

    Notes:
        Deprecation Warning:
            This function is deprecated and will be removed in a future version.
            Use `get_pca`, `plot_pca_scree`, `plot_pca1vs2`, and `get_pca_topk`
            instead for more modular control over PCA analysis.

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
