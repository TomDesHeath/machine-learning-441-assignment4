"""
Multi-Class Gradient Boosted Decision Trees (GBDT) with Multinomial Cross-Entropy Loss,
Shrinkage Learning Rate, Subsampling, and Newton-Raphson Leaf Updates.
"""

import numpy as np


class RegressionNode:
    """
    Regression Tree Node for GBDT pseudo-residual fitting.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=0.0, mse=0.0, n_samples=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value              # Leaf output scalar
        self.mse = mse
        self.n_samples = n_samples

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class DecisionTreeRegressor:
    """
    Fast Regression Decision Tree targeting continuous pseudo-residuals with Newton-Raphson leaf updates.
    """

    def __init__(self, max_depth=3, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', n_bins=64):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_bins = n_bins
        self.root = None

    def fit(self, X, residuals, probs, n_classes, learning_rate):
        """
        Fit regression tree on pseudo-residuals and update leaf node outputs via Newton-Raphson line search.
        """
        X = np.asarray(X, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)
        probs = np.asarray(probs, dtype=np.float64)

        rng = np.random.RandomState(42)
        indices = np.arange(len(residuals))
        self.root = self._build_tree(X, residuals, probs, indices, depth=0, n_classes=n_classes, learning_rate=learning_rate, rng=rng)
        return self

    def _compute_leaf_value(self, residuals_sub, probs_sub, n_classes, learning_rate):
        """
        Newton-Raphson line search update for multi-class log-loss leaf value.
        """
        if len(residuals_sub) == 0:
            return 0.0

        r_sum = np.sum(residuals_sub)
        d_sum = np.sum(probs_sub * (1.0 - probs_sub))
        mult = (n_classes - 1.0) / float(n_classes)

        if d_sum > 1e-10:
            return float(learning_rate * mult * (r_sum / d_sum))
        else:
            return 0.0

    def _build_tree(self, X, residuals, probs, indices, depth, n_classes, learning_rate, rng):
        n_samples = len(indices)
        res_sub = residuals[indices]
        probs_sub = probs[indices]

        var_res = np.var(res_sub) if n_samples > 0 else 0.0

        if depth >= self.max_depth or n_samples < self.min_samples_split or var_res == 0.0:
            val = self._compute_leaf_value(res_sub, probs_sub, n_classes, learning_rate)
            return RegressionNode(value=val, mse=var_res, n_samples=n_samples)

        N, D = X.shape
        if self.max_features == 'sqrt':
            m_feat = max(1, int(np.sqrt(D)))
        elif isinstance(self.max_features, int):
            m_feat = min(self.max_features, D)
        elif isinstance(self.max_features, float):
            m_feat = max(1, int(self.max_features * D))
        else:
            m_feat = D

        feat_indices = rng.choice(D, size=m_feat, replace=False)

        best_reduction = -1.0
        best_feat = None
        best_thresh = None

        mean_res = np.mean(res_sub)
        parent_ss = np.sum((res_sub - mean_res) ** 2)

        for f in feat_indices:
            x_col = X[indices, f]
            uniques = np.unique(x_col)
            if len(uniques) <= self.n_bins:
                thresholds = uniques
            else:
                percentiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
                thresholds = np.percentile(x_col, percentiles)

            for thresh in thresholds:
                left_mask = x_col <= thresh
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                res_l = res_sub[left_mask]
                res_r = res_sub[~left_mask]

                ss_l = np.sum((res_l - np.mean(res_l)) ** 2)
                ss_r = np.sum((res_r - np.mean(res_r)) ** 2)

                variance_reduction = parent_ss - (ss_l + ss_r)
                if variance_reduction > best_reduction:
                    best_reduction = variance_reduction
                    best_feat = f
                    best_thresh = thresh

        if best_reduction <= 1e-9 or best_feat is None:
            val = self._compute_leaf_value(res_sub, probs_sub, n_classes, learning_rate)
            return RegressionNode(value=val, mse=var_res, n_samples=n_samples)

        x_feat = X[indices, best_feat]
        left_mask = x_feat <= best_thresh
        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]

        left_child = self._build_tree(X, residuals, probs, left_idx, depth + 1, n_classes, learning_rate, rng)
        right_child = self._build_tree(X, residuals, probs, right_idx, depth + 1, n_classes, learning_rate, rng)

        return RegressionNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child,
                              mse=var_res, n_samples=n_samples)

    def _predict_sample(self, node, x):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            preds[i] = self._predict_sample(self.root, X[i])
        return preds


class GradientBoostingClassifier:
    """
    Multi-Class Gradient Boosted Decision Trees (GBDT).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=5,
                 min_samples_leaf=2, subsample=0.8, random_state=42):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds (M).
        learning_rate : float
            Shrinkage learning rate (eta).
        max_depth : int
            Maximum depth of base regression trees.
        subsample : float
            Subsample ratio of training instances per boosting round.
        random_state : int
            Random seed.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

        self.trees = []  # List of lists: trees[m][c] for boosting round m and class c
        self.n_classes_ = 0
        self.init_logits_ = None

    def _softmax(self, logits):
        """
        Numerically stable softmax transformation across classes.
        """
        max_l = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        Fit Multi-Class GBDT model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        N, D = X.shape
        self.n_classes_ = int(np.max(y)) + 1 if len(y) > 0 else 10

        # One-hot target matrix Y_oh
        Y_oh = np.zeros((N, self.n_classes_), dtype=np.float64)
        Y_oh[np.arange(N), y] = 1.0

        # Initialize log-odds logits
        self.init_logits_ = np.zeros((1, self.n_classes_), dtype=np.float64)
        F = np.repeat(self.init_logits_, N, axis=0)

        rng = np.random.RandomState(self.random_state)
        self.trees = []

        for m in range(self.n_estimators):
            probs = self._softmax(F)
            residuals = Y_oh - probs

            if self.subsample < 1.0:
                sub_size = max(1, int(self.subsample * N))
                sub_idx = rng.choice(N, size=sub_size, replace=False)
            else:
                sub_idx = np.arange(N)

            round_trees = []
            for c in range(self.n_classes_):
                reg_tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features='sqrt'
                )
                reg_tree.fit(X[sub_idx], residuals[sub_idx, c], probs[sub_idx, c], self.n_classes_, self.learning_rate)

                # Update current logits for class c across all training samples
                pred_c = reg_tree.predict(X)
                F[:, c] += pred_c
                round_trees.append(reg_tree)

            self.trees.append(round_trees)

        return self

    def predict_logits(self, X):
        """
        Compute continuous output logits F(x).
        """
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        F = np.repeat(self.init_logits_, N, axis=0)

        for round_trees in self.trees:
            for c in range(self.n_classes_):
                F[:, c] += round_trees[c].predict(X)

        return F

    def predict_proba(self, X):
        """
        Predict target class probabilities.
        """
        logits = self.predict_logits(X)
        return self._softmax(logits)

    def predict(self, X):
        """
        Predict target class labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
