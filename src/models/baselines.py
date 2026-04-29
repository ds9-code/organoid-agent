"""Trivial-but-strong baselines for each task."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from .base import BaselineModel


# -------------- T1: next-timepoint pseudobulk --------------

class IdentityNextTimepoint(BaselineModel):
    """y_pred = x_t. The target-to-beat baseline at small Δt."""
    name = "identity"

    def predict(self, X):
        return X


class PopulationMeanNextTimepoint(BaselineModel):
    """y_pred = mean(y_train). Sanity check: 'no input matters'."""
    name = "pop_mean"

    def fit(self, X, y):
        self.mean_ = y.mean(axis=0, keepdims=True)
        return self

    def predict(self, X):
        return np.broadcast_to(self.mean_, (X.shape[0], self.mean_.shape[1])).copy()


class LinearNextTimepoint(BaselineModel):
    """Per-gene Ridge regression."""
    name = "linear"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# -------------- T2: cell-type classification --------------

class LogRegCellType(BaselineModel):
    name = "logreg"

    def __init__(self, max_iter: int = 200, C: float = 1.0):
        self.max_iter = max_iter
        self.C = C

    def fit(self, X, y):
        self.model_ = LogisticRegression(
            max_iter=self.max_iter, C=self.C, n_jobs=-1, solver="lbfgs"
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class KNNCellType(BaselineModel):
    name = "knn"

    def __init__(self, k: int = 15):
        self.k = k

    def fit(self, X, y):
        self.model_ = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# -------------- T5: pseudotime --------------

class PC1Pseudotime(BaselineModel):
    """Use first principal component (sign-aligned to age_days) as pseudotime."""
    name = "pc1"

    def fit(self, X, y):
        self.pca_ = PCA(n_components=1).fit(X)
        z = self.pca_.transform(X).ravel()
        # align sign to true labels
        self.sign_ = 1.0 if np.corrcoef(z, y)[0, 1] >= 0 else -1.0
        return self

    def predict(self, X):
        return self.sign_ * self.pca_.transform(X).ravel()
