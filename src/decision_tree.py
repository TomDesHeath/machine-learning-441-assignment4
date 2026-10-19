"""
From-Scratch Decision Tree Classifier (CART) supporting Gini Impurity, Entropy, Cost-Sensitive Class Weighting,
Random Subspace Feature Selection, and Histogram-Accelerated Split Evaluation.
"""

import numpy as np


class Node:
    """
    Decision Tree Node representation.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, impurity=0.0, n_samples=0):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for split (X[:, feature] <= threshold)
        self.left = left                # Left child Node (X <= threshold)
        self.right = right              # Right child Node (X > threshold)
        self.value = value              # Probability distribution array over classes (leaf nodes)
        self.impurity = impurity        # Node Gini impurity or entropy
        self.n_samples = n_samples      # Number of samples reaching node

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    """
    CART Decision Tree Classifier for Multi-Class Classification.
    """

    def __init__(self, criterion='gini', max_depth=15, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, class_weight=None, random_state=None, n_bins=128):
        """
        Parameters:
        -----------
        criterion : str ('gini' or 'entropy')
            Splitting criterion.
        max_depth : int or None
            Maximum depth of decision tree.
        min_samples_split : int
            Minimum number of samples required to split an internal node.
        min_samples_leaf : int
            Minimum number of samples required at a leaf node.
        max_features : int, float, str ('sqrt', 'log2') or None
            Number of features to consider when looking for the best split.
        class_weight : str ('balanced') or dict or None
            Weights associated with classes.
        random_state : int or None
            Random seed for feature sampling reproducibility.
        n_bins : int
            Number of histogram quantile bins for accelerated continuous feature split evaluation.
        """
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else 999
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_bins = n_bins

        self.root = None
        self.n_classes_ = 0
        self.n_features_ = 0
        self.feature_importances_ = None
        self.class_weights_ = None

    def _compute_class_weights(self, y):
        """
        Compute sample weights based on target class frequencies.
        """
        classes, counts = np.unique(y, return_counts=True)
        self.n_classes_ = int(np.max(classes)) + 1 if len(classes) > 0 else 10
        weights = np.ones(self.n_classes_, dtype=np.float64)

        if self.class_weight == 'balanced':
            total_samples = len(y)
            for c, cnt in zip(classes, counts):
                weights[c] = total_samples / (len(classes) * cnt)
        elif isinstance(self.class_weight, dict):
            for c, w in self.class_weight.items():
                weights[c] = float(w)

        return weights

    def _impurity(self, y_weights):
        """
        Compute weighted Gini impurity or entropy for class weight totals.
        """
        total_w = np.sum(y_weights)
        if total_w <= 0.0:
            return 0.0

        p = y_weights / total_w
        p = p[p > 0]

        if self.criterion == 'gini':
            return 1.0 - np.sum(p ** 2)
        elif self.criterion == 'entropy':
            return -np.sum(p * np.log2(p))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _get_max_features_count(self):
        """
        Determine number of features to evaluate per split.
        """
        if self.max_features is None:
            return self.n_features_
        elif isinstance(self.max_features, int):
            return min(self.max_features, self.n_features_)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features_))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(self.n_features_)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(self.n_features_)))
        else:
            return self.n_features_

    def fit(self, X, y):
        """
        Build decision tree classifier from training set (X, y).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        self.n_features_ = X.shape[1]
        self.class_weights_ = self._compute_class_weights(y)
        self.feature_importances_ = np.zeros(self.n_features_, dtype=np.float64)

        rng = np.random.RandomState(self.random_state)
        
        # Build tree recursively
        self.root = self._build_tree(X, y, depth=0, rng=rng)

        # Normalize feature importances
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        return self

    def _build_tree(self, X, y, depth, rng):
        """
        Recursively construct tree nodes.
        """
        n_samples, n_feats = X.shape

        # Compute current node class weight vector and distribution
        y_counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
        weighted_counts = y_counts * self.class_weights_
        total_weight = np.sum(weighted_counts)

        leaf_prob = weighted_counts / total_weight if total_weight > 0 else np.full(self.n_classes_, 1.0 / self.n_classes_)
        current_impurity = self._impurity(weighted_counts)

        # Check stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            current_impurity == 0.0 or
            np.unique(y).size <= 1):
            return Node(value=leaf_prob, impurity=current_impurity, n_samples=n_samples)

        # Select random feature subset
        m_feat = self._get_max_features_count()
        feat_indices = rng.choice(n_feats, size=m_feat, replace=False)

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        # Evaluate best split over selected features
        for f in feat_indices:
            x_col = X[:, f]
            # Quantile binning for rapid threshold selection on continuous features
            if len(np.unique(x_col)) <= self.n_bins:
                thresholds = np.unique(x_col)
            else:
                percentiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
                thresholds = np.percentile(x_col, percentiles)

            for thresh in thresholds:
                left_mask = x_col <= thresh
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_counts = np.bincount(y[left_mask], minlength=self.n_classes_).astype(np.float64) * self.class_weights_
                right_counts = weighted_counts - left_counts

                w_left = np.sum(left_counts)
                w_right = np.sum(right_counts)

                if w_left <= 0 or w_right <= 0:
                    continue

                imp_left = self._impurity(left_counts)
                imp_right = self._impurity(right_counts)

                info_gain = total_weight * current_impurity - (w_left * imp_left + w_right * imp_right)

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feat = f
                    best_thresh = thresh

        if best_gain <= 1e-9 or best_feat is None:
            return Node(value=leaf_prob, impurity=current_impurity, n_samples=n_samples)

        # Record feature importance gain
        self.feature_importances_[best_feat] += best_gain

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1, rng)

        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child,
                    impurity=current_impurity, n_samples=n_samples)

    def _predict_sample_proba(self, node, x):
        """
        Traverse node tree for single sample prediction.
        """
        if node.is_leaf:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample_proba(node.left, x)
        else:
            return self._predict_sample_proba(node.right, x)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples X.
        """
        X = np.asarray(X, dtype=np.float64)
        probs = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        for i in range(X.shape[0]):
            probs[i, :] = self._predict_sample_proba(self.root, X[i])
        return probs

    def predict(self, X):
        """
        Predict target class labels for samples X.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
