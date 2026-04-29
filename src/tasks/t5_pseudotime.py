"""T5 — Pseudotime / real-time consistency."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import anndata as ad
from scipy.stats import spearmanr, kendalltau

from .base import Task


class PseudotimeTask(Task):
    name = "t5_pseudotime"

    def __init__(self, hvg_top: int | None = None):
        self.hvg_top = hvg_top

    def prepare(self, adata: ad.AnnData) -> Tuple:
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X = np.log1p(X.astype(np.float32))
        if self.hvg_top is not None and self.hvg_top < X.shape[1]:
            var = X.var(axis=0)
            keep = np.argsort(-var)[: self.hvg_top]
            X = X[:, keep]
        y = adata.obs["age_days"].to_numpy(dtype=float)
        # Train/eval = same matrix; the "model" returns pseudotimes for X
        return X, y, X, y

    def score(self, y_pred: np.ndarray, y_eval: np.ndarray):
        rho, _ = spearmanr(y_pred, y_eval)
        tau, _ = kendalltau(y_pred, y_eval)
        return {"spearman": float(rho), "kendall_tau": float(tau)}
