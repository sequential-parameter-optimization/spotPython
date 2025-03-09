import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from spotpython.gp.gp_sep import GPsep
import copy


class TreeGP(BaseEstimator, RegressorMixin):
    """A tree-based Gaussian Process model that partitions the input space
    and fits local GP models on each partition.

    This model divides the input space using clustering, trains a separate
    GPsep model on each cluster, and combines their predictions to cover
    the whole input space.

    Attributes:
        n_clusters: Number of clusters to partition the input space.
        min_points: Minimum number of points required in each cluster.
        weighting: Method for combining predictions ('hard' or 'distance').
        distance_power: Power parameter for distance-based weighting.
        gp_params: Parameters passed to each local GPsep model.
        cluster_model: The clustering algorithm used for partitioning.
        local_models: List of GPsep models for each partition.
        cluster_centers: Centers of each cluster.
        X_bounds: The bounds of the input space from training data.
        y_bounds: The bounds of the output space from training data.
    """

    def __init__(self, n_clusters=3, min_points=10, weighting="distance", distance_power=2, gp_params=None):
        """Initialize the TreeGP model with partitioning and GP parameters.

        Args:
            n_clusters: Number of clusters to create. Defaults to 3.
            min_points: Minimum points required per cluster. If a cluster has
                fewer points, it will be merged with nearest cluster.
                Defaults to 10.
            weighting: Method for combining predictions ('hard' or 'distance').
                Defaults to 'distance'.
            distance_power: Power parameter for distance-based weighting.
                Higher values make the weighting more local. Defaults to 2.
            gp_params: Dictionary of parameters to pass to each GPsep model.
                Defaults to None.
        """
        self.n_clusters = n_clusters
        self.min_points = min_points
        self.weighting = weighting
        self.distance_power = distance_power
        self.gp_params = gp_params or {}

        # These will be set during fitting
        self.cluster_model = None
        self.local_models = []
        self.cluster_centers = None
        self.X_bounds = None
        self.y_bounds = None
        self.cluster_indices = None

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Args:
            deep: If True, will return the parameters for this estimator and
                contained subobjects that are estimators. Defaults to True.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = {
            "n_clusters": self.n_clusters,
            "min_points": self.min_points,
            "weighting": self.weighting,
            "distance_power": self.distance_power,
            "gp_params": self.gp_params,
        }

        if deep and self.gp_params:
            gp_params_copy = copy.deepcopy(self.gp_params)
            params["gp_params"] = gp_params_copy

        return params

    def set_params(self, **parameters):
        """Set the parameters of this estimator.

        Args:
            **parameters: Estimator parameters as keyword arguments.

        Returns:
            self: Estimator instance.
        """
        for parameter, value in parameters.items():
            if parameter == "gp_params" and isinstance(value, dict):
                self.gp_params = copy.deepcopy(value)
            else:
                setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """Fit the TreeGP model by partitioning the input space and
        fitting local GP models to each partition.

        Args:
            X: Training input samples of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            self: Returns self.

        Raises:
            ValueError: If the number of samples is less than n_clusters or min_points.
        """
        # Convert pandas objects to numpy arrays if necessary
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        y = y.reshape(-1)
        n_samples = X.shape[0]

        if n_samples < max(self.n_clusters, self.min_points):
            raise ValueError(f"Number of samples ({n_samples}) must be at least " f"max(n_clusters, min_points) = {max(self.n_clusters, self.min_points)}")

        # Store data bounds for later normalization
        self.X_bounds = (np.min(X, axis=0), np.max(X, axis=0))
        self.y_bounds = (np.min(y), np.max(y))

        # Partition the input space using KMeans clustering
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")

        cluster_labels = self.cluster_model.fit_predict(X)
        self.cluster_centers = self.cluster_model.cluster_centers_

        # Handle clusters with too few points by merging them
        valid_clusters = []
        clusters_to_merge = []

        for i in range(self.n_clusters):
            cluster_size = np.sum(cluster_labels == i)
            if cluster_size >= self.min_points:
                valid_clusters.append(i)
            else:
                clusters_to_merge.append(i)

        # If there are clusters to merge, reassign their points to nearest valid cluster
        if clusters_to_merge:
            if not valid_clusters:
                # If no valid clusters, just use a single GPsep model
                print("No valid clusters with enough points. Falling back to single model.")
                self.n_clusters = 1
                self.cluster_centers = np.mean(X, axis=0, keepdims=True)
                cluster_labels = np.zeros(n_samples, dtype=int)
            else:
                # Reassign points from small clusters to nearest valid cluster
                nn = NearestNeighbors(n_neighbors=1).fit(self.cluster_centers[valid_clusters])

                for i in clusters_to_merge:
                    # Find points in this cluster
                    mask = cluster_labels == i
                    if np.any(mask):
                        # Find nearest valid cluster for each point
                        _, indices = nn.kneighbors(X[mask])
                        # Reassign to nearest valid cluster
                        cluster_labels[mask] = [valid_clusters[idx[0]] for idx in indices]

        # Save cluster assignments
        self.cluster_indices = [np.where(cluster_labels == i)[0] for i in range(self.n_clusters)]

        # Fit local GPsep models for each cluster
        self.local_models = []

        for i in range(self.n_clusters):
            idx = self.cluster_indices[i]
            if len(idx) > 0:  # Make sure we have points in this cluster
                # Create and fit a GPsep model for this cluster
                gp = GPsep(**self.gp_params)
                gp.fit(X[idx], y[idx])
                self.local_models.append(gp)
            else:
                # This shouldn't happen after our merging step, but just in case
                self.local_models.append(None)

        return self

    def predict(self, X, return_std=False, **kwargs):
        """Predict using the TreeGP model.

        Args:
            X: Query points of shape (n_samples, n_features).
            return_std: Whether to return standard deviations. Defaults to False.
            **kwargs: Additional keyword arguments passed to the local models' predict method.

        Returns:
            If return_std is False, returns predicted values.
            If return_std is True, returns (predicted_values, standard_deviations).
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        if return_std:
            y_std = np.zeros(n_samples)

        # Get distances to cluster centers for weighting
        distances = np.zeros((n_samples, self.n_clusters))

        for i in range(self.n_clusters):
            # Calculate Euclidean distance to each cluster center
            diff = X - self.cluster_centers[i]
            distances[:, i] = np.sqrt(np.sum(diff**2, axis=1))

        if self.weighting == "hard":
            # Hard assignment: use closest cluster's model for prediction
            cluster_assignments = np.argmin(distances, axis=1)

            for i in range(self.n_clusters):
                mask = cluster_assignments == i
                if np.any(mask) and self.local_models[i] is not None:
                    if return_std:
                        # Use return_full=True to get the full prediction dictionary
                        preds = self.local_models[i].predict(X[mask], return_full=True)
                        y_pred[mask] = preds["mean"]

                        # Extract standard deviations from the covariance matrix
                        if "Sigma" in preds:
                            # If full covariance matrix is returned, extract diagonal
                            if len(preds["Sigma"].shape) == 2:
                                y_std[mask] = np.sqrt(np.diag(preds["Sigma"]))
                            else:
                                # If already diagonal elements, just use directly
                                y_std[mask] = np.sqrt(preds["Sigma"])
                        elif "s2" in preds:
                            # Some implementations return variances as "s2"
                            y_std[mask] = np.sqrt(preds["s2"])
                    else:
                        y_pred[mask] = self.local_models[i].predict(X[mask], return_full=False)

        else:  # distance weighting
            # Compute weights from distances (closer = higher weight)
            weights = 1.0 / np.maximum(distances**self.distance_power, 1e-10)
            # Normalize weights
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            weights = weights / np.maximum(weights_sum, 1e-10)

            # Get predictions from each model and combine with weights
            all_preds = np.zeros((n_samples, self.n_clusters))
            all_var = np.zeros((n_samples, self.n_clusters))  # Store variances instead of std devs

            for i in range(self.n_clusters):
                if self.local_models[i] is not None:
                    if return_std:
                        # Use return_full=True to get the full prediction dictionary
                        preds = self.local_models[i].predict(X, return_full=True)
                        all_preds[:, i] = preds["mean"]

                        # Extract variances from the covariance matrix
                        if "Sigma" in preds:
                            # If full covariance matrix is returned, extract diagonal
                            if len(preds["Sigma"].shape) == 2:
                                all_var[:, i] = np.diag(preds["Sigma"])
                            else:
                                # If already diagonal elements, use directly
                                all_var[:, i] = preds["Sigma"]
                        elif "s2" in preds:
                            all_var[:, i] = preds["s2"]
                    else:
                        all_preds[:, i] = self.local_models[i].predict(X)

            # Combine predictions using weights
            y_pred = np.sum(all_preds * weights, axis=1)

            if return_std:
                # Combine variances (with appropriate weighting)
                y_var = np.sum(all_var * (weights**2), axis=1)
                y_std = np.sqrt(y_var)

        if return_std:
            return y_pred, y_std
        else:
            return y_pred

    def score(self, X, y, **kwargs):
        """Calculate the coefficient of determination R^2 of the prediction.

        Args:
            X: Test input samples.
            y: True values for X.
            **kwargs: Additional arguments passed to predict().

        Returns:
            float: R^2 score.
        """
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X, **kwargs))


# Helper function to create and fit a TreeGP model in a single call
def newTreeGP(X, y, n_clusters=3, **kwargs):
    """
    Create and fit a TreeGP model in a single function call.

    Args:
        X: Training input samples.
        y: Target values.
        n_clusters: Number of clusters to partition the data. Defaults to 3.
        **kwargs: Additional parameters passed to TreeGP constructor.

    Returns:
        TreeGP: The fitted TreeGP model.
    """
    model = TreeGP(n_clusters=n_clusters, **kwargs)
    return model.fit(X, y)
