"""
Heterogeneous Stacking Ensemble Meta-Learner blending out-of-fold class probability predictions
from k-NN, Decision Trees, Random Forests, and Gradient Boosted Decision Trees.
"""

import numpy as np


class MetaLogisticRegression:
    """
    Multinomial Logistic Regression Meta-Learner with L2 Regularization (SGD / Gradient Descent).
    """

    def __init__(self, C=1.0, max_iter=200, lr=0.1, random_state=42):
        self.C = C
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None
        self.n_classes_ = 0

    def _softmax(self, logits):
        max_l = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        N, D = X.shape
        self.n_classes_ = int(np.max(y)) + 1

        Y_oh = np.zeros((N, self.n_classes_), dtype=np.float64)
        Y_oh[np.arange(N), y] = 1.0

        rng = np.random.RandomState(self.random_state)
        self.weights_ = rng.normal(0, 0.01, size=(D, self.n_classes_))
        self.bias_ = np.zeros((1, self.n_classes_), dtype=np.float64)

        reg_lambda = 1.0 / (self.C * N) if self.C > 0 else 0.0

        for it in range(self.max_iter):
            logits = np.dot(X, self.weights_) + self.bias_
            probs = self._softmax(logits)
            error = probs - Y_oh

            grad_w = np.dot(X.T, error) / N + reg_lambda * self.weights_
            grad_b = np.sum(error, axis=0, keepdims=True) / N

            self.weights_ -= self.lr * grad_w
            self.bias_ -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        logits = np.dot(X, self.weights_) + self.bias_
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


import inspect

def _clone_estimator(model):
    cls = model.__class__
    sig = inspect.signature(cls.__init__)
    params = {}
    for param_name in sig.parameters.keys():
        if param_name != 'self' and hasattr(model, param_name):
            params[param_name] = getattr(model, param_name)
    return cls(**params)


class StackingClassifier:
    """
    Two-Tier Stacking Ensemble Classifier.
    """

    def __init__(self, estimators, final_estimator=None, cv=5, random_state=42):
        """
        Parameters:
        -----------
        estimators : list of tuple (name, estimator_instance)
            Base Level-0 model instances.
        final_estimator : estimator instance or None
            Level-1 Meta-Learner (defaults to MetaLogisticRegression).
        cv : int
            Number of cross-validation folds for out-of-fold prediction generation.
        random_state : int
            Random seed.
        """
        self.estimators = estimators
        self.final_estimator = final_estimator if final_estimator is not None else MetaLogisticRegression(C=1.0)
        self.cv = cv
        self.random_state = random_state
        self.fitted_estimators_ = []
        self.n_classes_ = 0

    def _stratified_cv_splits(self, y):
        """
        Generate stratified cross-validation index splits.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)

        class_indices = {c: np.where(y == c)[0] for c in classes}
        for c in classes:
            rng.shuffle(class_indices[c])

        folds = [[] for _ in range(self.cv)]
        for c in classes:
            c_idx = class_indices[c]
            split_sizes = np.full(self.cv, len(c_idx) // self.cv)
            split_sizes[: len(c_idx) % self.cv] += 1

            current = 0
            for k, size in enumerate(split_sizes):
                folds[k].extend(c_idx[current : current + size])
                current += size

        splits = []
        all_idx = np.arange(n_samples)
        for k in range(self.cv):
            val_idx = np.array(folds[k], dtype=np.int64)
            train_idx = np.setdiff1d(all_idx, val_idx)
            splits.append((train_idx, val_idx))

        return splits

    def fit(self, X, y):
        """
        Fit Level-0 base learners on CV splits, construct out-of-fold feature matrix Z, fit Level-1 meta-learner,
        and re-fit Level-0 base learners on 100% of data.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        N, D = X.shape
        self.n_classes_ = int(np.max(y)) + 1
        K_models = len(self.estimators)

        # Matrix for Level-1 meta features
        Z_oof = np.zeros((N, K_models * self.n_classes_), dtype=np.float64)

        cv_splits = self._stratified_cv_splits(y)

        # Generate out-of-fold predictions
        for m_idx, (name, base_model) in enumerate(self.estimators):
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val = X[val_idx]

                # Instantiate new model clone with same parameters
                model_clone = _clone_estimator(base_model)
                model_clone.fit(X_tr, y_tr)

                val_probs = model_clone.predict_proba(X_val)

                col_start = m_idx * self.n_classes_
                col_end = (m_idx + 1) * self.n_classes_
                Z_oof[val_idx, col_start:col_end] = val_probs

        # Fit Level-1 Meta-Learner on out-of-fold probability predictions Z_oof
        self.final_estimator.fit(Z_oof, y)

        # Re-fit Level-0 Base Learners on 100% training data for inference
        self.fitted_estimators_ = []
        for name, base_model in self.estimators:
            model_full = _clone_estimator(base_model)
            model_full.fit(X, y)
            self.fitted_estimators_.append((name, model_full))

        return self

    def _transform_meta_features(self, X):
        """
        Construct meta-features Z for test set evaluation.
        """
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        K_models = len(self.fitted_estimators_)
        Z_test = np.zeros((N, K_models * self.n_classes_), dtype=np.float64)

        for m_idx, (name, model_full) in enumerate(self.fitted_estimators_):
            probs = model_full.predict_proba(X)
            col_start = m_idx * self.n_classes_
            col_end = (m_idx + 1) * self.n_classes_
            Z_test[:, col_start:col_end] = probs

        return Z_test

    def predict_proba(self, X):
        """
        Predict target class probabilities via Stacking meta-learner.
        """
        Z_test = self._transform_meta_features(X)
        return self.final_estimator.predict_proba(Z_test)

    def predict(self, X):
        """
        Predict target class labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
