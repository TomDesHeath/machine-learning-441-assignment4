"""
Vectorized k-Nearest Neighbours (k-NN) Classifier for High-Dimensional Network Telemetry.
"""

import numpy as np


class KNearestNeighbours:
    """
    k-Nearest Neighbours Classifier with Vectorized Distance Queries and Distance-Weighted Voting.
    """

    def __init__(self, n_neighbors=5, weights='distance', p=2, batch_size=5000):
        """
        Parameters:
        -----------
        n_neighbors : int
            Number of nearest neighbours (k).
        weights : str ('uniform' or 'distance')
            Weight function used in prediction.
        p : int
            Power parameter for Minkowski metric (p=2 is Euclidean).
        batch_size : int
            Chunk size for distance matrix evaluation to optimize memory footprint.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.n_classes_ = 0

    def fit(self, X, y):
        """
        Fit k-NN model by storing reference training set.
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(np.max(self.y_train)) + 1
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using k nearest neighbours.
        """
        X = np.asarray(X, dtype=np.float64)
        N_test = X.shape[0]
        N_train = self.X_train.shape[0]
        probs = np.zeros((N_test, self.n_classes_), dtype=np.float64)

        train_sq_norms = np.sum(self.X_train ** 2, axis=1)

        # Process test instances in chunks
        for start_idx in range(0, N_test, self.batch_size):
            end_idx = min(start_idx + self.batch_size, N_test)
            X_batch = X[start_idx:end_idx]

            if self.p == 2:
                # Fast Euclidean distance calculation via dot products: ||A-B||^2 = ||A||^2 + ||B||^2 - 2AB^T
                batch_sq_norms = np.sum(X_batch ** 2, axis=1, keepdims=True)
                dot_prod = np.dot(X_batch, self.X_train.T)
                dist_sq = batch_sq_norms + train_sq_norms - 2.0 * dot_prod
                dist_sq = np.maximum(dist_sq, 0.0)
                dists = np.sqrt(dist_sq)
            else:
                # General Minkowski distance calculation
                dists = np.zeros((X_batch.shape[0], N_train), dtype=np.float64)
                for b in range(X_batch.shape[0]):
                    dists[b, :] = np.sum(np.abs(self.X_train - X_batch[b]) ** self.p, axis=1) ** (1.0 / self.p)

            # Partition top k smallest distances
            knn_indices = np.argpartition(dists, self.n_neighbors, axis=1)[:, :self.n_neighbors]

            for b in range(X_batch.shape[0]):
                indices = knn_indices[b]
                d = dists[b, indices]

                # Sort selected k neighbours
                sort_ord = np.argsort(d)
                indices = indices[sort_ord]
                d = d[sort_ord]

                labels = self.y_train[indices]

                if self.weights == 'distance':
                    # Distance-weighted voting with epsilon protection
                    w = 1.0 / (d + 1e-8)
                else:
                    w = np.ones_like(d, dtype=np.float64)

                total_w = np.sum(w)
                for lab, weight in zip(labels, w):
                    probs[start_idx + b, lab] += weight

                if total_w > 0:
                    probs[start_idx + b, :] /= total_w

        return probs

    def predict(self, X):
        """
        Predict target class labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
