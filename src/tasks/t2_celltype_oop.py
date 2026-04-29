"""T2 — Out-of-protocol cell-type prediction."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder

from .base import Task
from ..eval.metrics import macro_f1, accuracy


class CellTypeOOPTask(Task):
    name = "t2_celltype_oop"

    def __init__(self, holdout_protocols=None, holdout_frac: float = 0.25,
                 label_col: str = "cell_type", protocol_col: str = "protocol",
                 random_state: int = 0):
        self.holdout_protocols = holdout_protocols
        self.holdout_frac = holdout_frac
        self.label_col = label_col
        self.protocol_col = protocol_col
        self.random_state = random_state
        self.classes_ = None

    def prepare(self, adata: ad.AnnData) -> Tuple:
        rng = np.random.default_rng(self.random_state)
        all_protocols = adata.obs[self.protocol_col].unique().tolist()
        if self.holdout_protocols is None:
            n_hold = max(1, int(len(all_protocols) * self.holdout_frac))
            rng.shuffle(all_protocols)
            holdout = set(all_protocols[:n_hold])
        else:
            holdout = set(self.holdout_protocols)

        is_eval = adata.obs[self.protocol_col].isin(holdout).to_numpy()
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        # log1p normalisation, simple
        X = np.log1p(X.astype(np.float32))

        le = LabelEncoder().fit(adata.obs[self.label_col].astype(str))
        y = le.transform(adata.obs[self.label_col].astype(str))
        self.classes_ = le.classes_

        return X[~is_eval], y[~is_eval], X[is_eval], y[is_eval]

    def score(self, y_pred: np.ndarray, y_eval: np.ndarray):
        return {
            "accuracy": float(accuracy(y_pred, y_eval)),
            "macro_f1": float(macro_f1(y_pred, y_eval)),
        }
