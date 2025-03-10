import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from spotpython.gp.gp_sep import GPsep
import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


class TreeGP(BaseEstimator, RegressorMixin):
    """A tree-based Gaussian Process model that partitions the input space
    using regression trees and fits local GP models on each partition.

    This model uses a decision tree to create regions based on both X and y values,
    then trains separate GPsep models on each region. Model predictions are blended
    along boundaries for smoothness.

    Attributes:
        max_depth: Maximum depth of the regression tree for partitioning.
        min_samples_leaf: Minimum samples required in each leaf node.
        smooth_transitions: Whether to use soft transitions between regions.
        smooth_factor: Controls the smoothness of transitions (higher = smoother).
        gp_params: Parameters passed to each local GPsep model.
        tree_model: The regression tree used for partitioning.
        local_models: Dictionary mapping leaf indices to GPsep models.
        leaf_stats: Statistics for each leaf node.
        scaler_X: Feature scaler for input normalization.
        scaler_y: Response scaler for output normalization.
    """

    def __init__(self, max_depth=3, min_samples_leaf=10, smooth_transitions=True, smooth_factor=2.0, gp_params=None, auto_scale=True, plot_partitions=False):
        """Initialize the TreeGP model with regression tree partitioning.

        Args:
            max_depth: Maximum depth of the tree for partitioning. Defaults to 3.
            min_samples_leaf: Minimum samples required in each leaf node. Defaults to 10.
            smooth_transitions: Whether to use soft transitions between regions.
                Defaults to True.
            smooth_factor: Controls smoothness of transitions (higher = smoother).
                Defaults to 2.0.
            gp_params: Dictionary of parameters to pass to each GPsep model.
                Defaults to None.
            auto_scale: Whether to automatically scale inputs and outputs.
                Defaults to True.
            plot_partitions: Whether to plot the partitions after fitting.
                Defaults to False.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.smooth_transitions = smooth_transitions
        self.smooth_factor = smooth_factor
        self.gp_params = gp_params or {}
        self.auto_scale = auto_scale
        self.plot_partitions = plot_partitions

        # These will be set during fitting
        self.tree_model = None
        self.local_models = {}
        self.leaf_stats = {}
        self.scaler_X = None
        self.scaler_y = None
        self.leaves = None
        self.X_train = None
        self.y_train = None

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Args:
            deep: If True, will return the parameters for this estimator and
                contained subobjects that are estimators. Defaults to True.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "smooth_transitions": self.smooth_transitions,
            "smooth_factor": self.smooth_factor,
            "gp_params": self.gp_params,
            "auto_scale": self.auto_scale,
            "plot_partitions": self.plot_partitions,
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

    def _find_leaves_and_samples(self, tree, n_samples):
        """Extract leaf node indices and corresponding sample indices from the fitted tree.

        Args:
            tree: The fitted decision tree.
            n_samples: Number of samples in training data.

        Returns:
            dict: Mapping from leaf index to array of sample indices.
        """
        # Get the decision path for all samples
        node_indicator = tree.decision_path(self.X_train)

        # Get the leaf node for each sample
        leaf_ids = tree.apply(self.X_train)

        # Create a dictionary mapping each leaf node to its samples
        leaves = {}
        for sample_id in range(n_samples):
            leaf_id = leaf_ids[sample_id]
            if leaf_id not in leaves:
                leaves[leaf_id] = []
            leaves[leaf_id].append(sample_id)

        # Convert lists to numpy arrays
        for leaf_id in leaves:
            leaves[leaf_id] = np.array(leaves[leaf_id])

        return leaves

    def fit(self, X, y):
        """Fit the TreeGP model by partitioning the input space with a regression tree
        and fitting local GP models to each partition.

        Args:
            X: Training input samples of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            self: Returns self.
        """
        # Convert pandas objects to numpy arrays if necessary
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        y = y.reshape(-1)
        n_samples, n_features = X.shape

        # Store original data for later use
        self.X_train = X
        self.y_train = y

        # Scale data if requested
        if self.auto_scale:
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(X)

            self.scaler_y = StandardScaler()
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            X_scaled = X
            y_scaled = y

        # Fit a regression tree to partition the input space
        self.tree_model = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=42)
        self.tree_model.fit(X_scaled, y_scaled)

        # Find leaf nodes and their corresponding samples
        self.leaves = self._find_leaves_and_samples(self.tree_model, n_samples)

        # Fit a GP model for each leaf node
        self.local_models = {}
        self.leaf_stats = {}

        for leaf_id, sample_indices in self.leaves.items():
            # Skip leaves with too few points (shouldn't happen with min_samples_leaf, but just in case)
            if len(sample_indices) < 3:  # Need at least 3 points for a reasonable GP
                continue

            # Get data for this leaf
            X_leaf = X[sample_indices]
            y_leaf = y[sample_indices]

            # Store statistics for this leaf
            self.leaf_stats[leaf_id] = {"mean": np.mean(y_leaf), "std": np.std(y_leaf), "center": np.mean(X_leaf, axis=0), "n_samples": len(sample_indices)}

            # Create and fit a GPsep model for this leaf
            gp = GPsep(**self.gp_params)
            gp.fit(X_leaf, y_leaf)
            self.local_models[leaf_id] = gp

        # Optionally visualize the partitions
        if self.plot_partitions and n_features <= 2:
            self.plot_tree_partitions()

        return self

    def _get_leaf_probabilities(self, X):
        """Calculate probabilities for each leaf node for test points.

        For each test point, find the leaf it belongs to and assign a probability
        of 1.0 to that leaf. If smooth_transitions is True, also assign non-zero
        probabilities to nearby leaves based on proximity.

        Args:
            X: Test points of shape (n_samples, n_features).

        Returns:
            array: Probabilities of shape (n_samples, n_leaves).
        """
        n_samples = X.shape[0]
        leaf_ids = list(self.local_models.keys())
        n_leaves = len(leaf_ids)

        # Scale input if needed
        if self.auto_scale:
            X_scaled = self.scaler_X.transform(X)
        else:
            X_scaled = X

        # Get leaf node assignments from the tree
        leaf_assignments = self.tree_model.apply(X_scaled)

        # Initialize probabilities matrix
        probabilities = np.zeros((n_samples, n_leaves))

        # Simple case: hard assignments
        if not self.smooth_transitions:
            for i, leaf_id in enumerate(leaf_assignments):
                try:
                    leaf_idx = leaf_ids.index(leaf_id)
                    probabilities[i, leaf_idx] = 1.0
                except ValueError:
                    # If leaf_id is not in local_models (rare edge case), use nearest leaf
                    dists = []
                    for lid in leaf_ids:
                        leaf_center = self.leaf_stats[lid]["center"]
                        dist = np.sum((X[i] - leaf_center) ** 2)
                        dists.append(dist)
                    nearest_leaf_idx = np.argmin(dists)
                    probabilities[i, nearest_leaf_idx] = 1.0

            return probabilities

        # More complex case: smooth transitions
        # We need to calculate distances to all leaf centers
        for i in range(n_samples):
            x = X[i]
            # Vector of distances to each leaf center
            distances = np.zeros(n_leaves)

            for j, leaf_id in enumerate(leaf_ids):
                leaf_center = self.leaf_stats[leaf_id]["center"]
                distances[j] = np.sqrt(np.sum((x - leaf_center) ** 2))

            # Convert distances to weights using a kernel function
            # Higher smooth_factor makes transitions smoother but may blur important boundaries
            bandwidth = np.mean(distances) / self.smooth_factor
            weights = np.exp(-0.5 * (distances / bandwidth) ** 2)

            # If the point is directly in a leaf, boost that leaf's weight
            try:
                leaf_idx = leaf_ids.index(leaf_assignments[i])
                weights[leaf_idx] *= 3.0  # Boost the weight of the assigned leaf
            except ValueError:
                pass  # Leaf not in our models, just use distances

            # Normalize weights to sum to 1
            if np.sum(weights) > 0:
                probabilities[i] = weights / np.sum(weights)
            else:
                # Fallback to nearest leaf
                nearest_leaf_idx = np.argmin(distances)
                probabilities[i, nearest_leaf_idx] = 1.0

        return probabilities

    def predict(self, X, return_std=False, **kwargs):
        """Predict using the TreeGP model, combining predictions from multiple local models.

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
        leaf_ids = list(self.local_models.keys())
        n_leaves = len(leaf_ids)

        # Get probability weights for each leaf
        leaf_probs = self._get_leaf_probabilities(X)

        # Get predictions from each local model
        all_preds = np.zeros((n_samples, n_leaves))
        all_vars = np.zeros((n_samples, n_leaves))

        for i, leaf_id in enumerate(leaf_ids):
            model = self.local_models[leaf_id]

            # Only predict for samples with non-zero weight for this leaf
            # (saves computation for distant points)
            mask = leaf_probs[:, i] > 1e-6
            if np.any(mask):
                # Get predictions
                preds = model.predict(X[mask], return_full=True)
                all_preds[mask, i] = preds["mean"]

                # Extract variances from the covariance matrix
                if "Sigma" in preds:
                    # If full covariance matrix is returned, extract diagonal
                    if len(preds["Sigma"].shape) == 2:
                        all_vars[mask, i] = np.diag(preds["Sigma"])
                    else:
                        # If already diagonal elements, use directly
                        all_vars[mask, i] = preds["Sigma"]
                elif "s2" in preds:
                    all_vars[mask, i] = preds["s2"]

        # Weight predictions by leaf probabilities
        y_pred = np.sum(all_preds * leaf_probs, axis=1)

        if return_std:
            # Weighted variance calculation (accounting for model uncertainty + transition uncertainty)

            # Base uncertainty: weighted average of variances
            weighted_vars = np.sum(all_vars * leaf_probs, axis=1)

            # Plus additional uncertainty from model disagreement
            means_diff = all_preds - y_pred.reshape(-1, 1)
            model_variance = np.sum(leaf_probs * (means_diff**2), axis=1)

            # Combine both sources of uncertainty
            total_var = weighted_vars + model_variance
            y_std = np.sqrt(total_var)

            return y_pred, y_std
        else:
            return y_pred

    def plot_tree_partitions(self, figsize=(12, 10), cmap="viridis"):
        """Visualize the tree partitions and local models.

        Works for 1D and 2D input spaces.

        Args:
            figsize: Figure size as (width, height). Defaults to (12, 10).
            cmap: Colormap to use. Defaults to 'viridis'.
        """
        X = self.X_train
        y = self.y_train
        n_features = X.shape[1]

        if n_features > 2:
            raise ValueError("Visualization only works for 1D or 2D inputs")

        leaf_ids = list(self.local_models.keys())
        n_leaves = len(leaf_ids)
        colors = plt.cm.tab10(np.linspace(0, 1, n_leaves))

        plt.figure(figsize=figsize)

        # 1D case
        if n_features == 1:
            # Plot original data points
            for i, leaf_id in enumerate(leaf_ids):
                indices = self.leaves[leaf_id]
                plt.scatter(X[indices], y[indices], color=colors[i], label=f"Leaf {i}", alpha=0.7, edgecolor="k")

            # Generate a grid for smooth predictions
            x_min, x_max = X.min(), X.max()
            margin = 0.1 * (x_max - x_min)
            X_grid = np.linspace(x_min - margin, x_max + margin, 1000).reshape(-1, 1)

            # Get predictions for the grid
            y_pred, y_std = self.predict(X_grid, return_std=True)

            # Plot predictions
            plt.plot(X_grid, y_pred, "r-", lw=2, label="TreeGP prediction")
            plt.fill_between(X_grid.ravel(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, color="red", alpha=0.2, label="95% confidence")

            # Plot local model predictions
            if self.smooth_transitions:
                leaf_probs = self._get_leaf_probabilities(X_grid)
                for i, leaf_id in enumerate(leaf_ids):
                    # Only show local model within its high-probability region
                    mask = leaf_probs[:, i] > 0.2
                    if np.any(mask):
                        model = self.local_models[leaf_id]
                        local_pred = model.predict(X_grid[mask])
                        plt.plot(X_grid[mask], local_pred, "--", color=colors[i], alpha=0.5, label=f"Local model {i}")

            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.title("TreeGP: Regression Tree Partitions with Local GP Models")

        # 2D case
        else:
            # Create a subplot grid: 3 rows, 2 columns
            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # Plot 1: Original data points colored by leaf assignment
            ax = axes[0, 0]
            for i, leaf_id in enumerate(leaf_ids):
                indices = self.leaves[leaf_id]
                ax.scatter(X[indices, 0], X[indices, 1], color=colors[i], label=f"Leaf {i}", alpha=0.7)
            ax.set_title("Training Data Partitions")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.legend()

            # Plot 2: Create a mesh grid for visualization
            ax = axes[0, 1]
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            grid = np.c_[xx.ravel(), yy.ravel()]

            # Get leaf assignments for the grid
            if self.auto_scale:
                grid_scaled = self.scaler_X.transform(grid)
                leaf_assignments = self.tree_model.apply(grid_scaled)
            else:
                leaf_assignments = self.tree_model.apply(grid)

            # Create a color map for visualization
            Z = np.zeros(leaf_assignments.shape)
            for i, leaf_id in enumerate(leaf_ids):
                Z[leaf_assignments == leaf_id] = i

            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
            ax.scatter(X[:, 0], X[:, 1], c="black", edgecolor="k", s=20)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_title("Decision Tree Regions")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

            # Plot 3: Prediction heatmap
            ax = axes[1, 0]
            grid_pred = self.predict(grid).reshape(xx.shape)

            # Plot the heatmap
            c = ax.contourf(xx, yy, grid_pred, 50, cmap="viridis")
            plt.colorbar(c, ax=ax)
            ax.set_title("TreeGP Predictions")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

            # Plot 4: Uncertainty heatmap
            ax = axes[1, 1]
            _, grid_std = self.predict(grid, return_std=True)
            grid_std = grid_std.reshape(xx.shape)

            c = ax.contourf(xx, yy, grid_std, 50, cmap="plasma")
            plt.colorbar(c, ax=ax)
            ax.set_title("Prediction Uncertainty (Std)")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

            plt.tight_layout()

        plt.show()


def newTreeGP(X, y, **kwargs):
    """
    Create and fit a TreeGP model in a single function call.

    Args:
        X: Training input samples.
        y: Target values.
        **kwargs: Additional parameters passed to TreeGP constructor.

    Returns:
        TreeGP: The fitted TreeGP model.
    """
    model = TreeGP(**kwargs)
    return model.fit(X, y)
