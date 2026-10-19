"""
Custom Random Forest Classifier supporting Bootstrap Aggregation, Random Subspace Feature Subsampling,
Out-of-Bag (OOB) Error Tracking, Mean Decrease in Impurity (MDI) Extraction, and Multi-Core Parallelization.
"""

import numpy as np
import concurrent.futures
from .decision_tree import DecisionTreeClassifier


def _train_single_tree(tree_params, X, y, sample_idx, seed):
    """
    Helper function for parallel training of an individual decision tree.
    """
    tree = DecisionTreeClassifier(**tree_params, random_state=seed)
    X_sample = X[sample_idx]
    y_sample = y[sample_idx]
    tree.fit(X_sample, y_sample)
    return tree


class RandomForestClassifier:
    """
    Bootstrap Aggregated Random Forest Ensemble Classifier.
    """

    def __init__(self, n_estimators=100, max_depth=15, min_samples_split=2, min_samples_leaf=1,
                 max_features='sqrt', bootstrap=True, class_weight='balanced', n_jobs=-1, random_state=42):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of decision trees in ensemble (M).
        max_depth : int or None
            Maximum depth of member decision trees.
        min_samples_split : int
            Minimum samples required to split a node.
        min_samples_leaf : int
            Minimum samples required at a leaf node.
        max_features : str ('sqrt', 'log2') or float or int
            Feature subsampling rule per split node.
        bootstrap : bool
            Whether to use bootstrap sampling for tree training.
        class_weight : str ('balanced') or dict or None
            Cost-sensitive class weighting mode.
        n_jobs : int
            Number of parallel workers (-1 uses all available CPU cores).
        random_state : int
            Random seed.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.trees = []
        self.n_classes_ = 0
        self.n_features_ = 0
        self.feature_importances_ = None
        self.oob_score_ = 0.0
        self.oob_probabilities_ = None

    def fit(self, X, y):
        """
        Build Random Forest ensemble of decision trees.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        N, D = X.shape
        self.n_features_ = D
        self.n_classes_ = int(np.max(y)) + 1 if len(y) > 0 else 10

        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 1000000, size=self.n_estimators)

        # Generate bootstrap sample indices and track OOB instances
        sample_indices = []
        oob_counts = np.zeros((N, self.n_classes_), dtype=np.float64)

        for m in range(self.n_estimators):
            if self.bootstrap:
                boot_idx = rng.choice(N, size=N, replace=True)
            else:
                boot_idx = np.arange(N)
            sample_indices.append(boot_idx)

        tree_params = {
            'criterion': 'gini',
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'class_weight': self.class_weight
        }

        # Train trees sequentially or in parallel
        self.trees = []
        if self.n_jobs == 1:
            for m in range(self.n_estimators):
                tree = _train_single_tree(tree_params, X, y, sample_indices[m], seeds[m])
                self.trees.append(tree)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=None if self.n_jobs == -1 else self.n_jobs) as executor:
                futures = [executor.submit(_train_single_tree, tree_params, X, y, sample_indices[m], seeds[m]) for m in range(self.n_estimators)]
                for future in concurrent.futures.as_completed(futures):
                    self.trees.append(future.result())

        # Aggregate feature importances across all member trees
        self.feature_importances_ = np.zeros(D, dtype=np.float64)
        for tree in self.trees:
            if tree.feature_importances_ is not None:
                self.feature_importances_ += tree.feature_importances_

        if self.n_estimators > 0:
            self.feature_importances_ /= len(self.trees)

        # Calculate Out-Of-Bag (OOB) predictions if bootstrap is enabled
        if self.bootstrap:
            oob_predictions = np.zeros((N, self.n_classes_), dtype=np.float64)
            oob_sample_counts = np.zeros(N, dtype=np.int64)

            for m, boot_idx in enumerate(sample_indices):
                tree = self.trees[m]
                in_bag = np.zeros(N, dtype=bool)
                in_bag[boot_idx] = True
                oob_mask = ~in_bag
                oob_idx_list = np.where(oob_mask)[0]

                if len(oob_idx_list) > 0:
                    preds_proba = tree.predict_proba(X[oob_idx_list])
                    oob_predictions[oob_idx_list] += preds_proba
                    oob_sample_counts[oob_idx_list] += 1

            valid_oob = oob_sample_counts > 0
            if np.any(valid_oob):
                oob_predictions[valid_oob] /= oob_sample_counts[valid_oob, None]
                self.oob_probabilities_ = oob_predictions
                oob_preds = np.argmax(oob_predictions[valid_oob], axis=1)
                self.oob_score_ = float(np.mean(oob_preds == y[valid_oob]))
            else:
                self.oob_score_ = 0.0

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities by averaging predictions across all trees.
        """
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        probs = np.zeros((N, self.n_classes_), dtype=np.float64)

        for tree in self.trees:
            probs += tree.predict_proba(X)

        if len(self.trees) > 0:
            probs /= len(self.trees)

        return probs

    def predict(self, X):
        """
        Predict target class labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
