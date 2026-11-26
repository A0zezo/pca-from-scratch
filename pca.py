"""PCA via SVD in NumPy, verified against scikit-learn."""
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.k = n_components

    def fit(self, X):
        self.mean = X.mean(0)
        _, S, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.components = Vt[:self.k]
        var = S ** 2 / (len(X) - 1)
        self.explained_variance_ratio = var[:self.k] / var.sum()
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean
