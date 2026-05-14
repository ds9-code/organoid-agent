"""T1 — Next-timepoint pseudobulk prediction."""
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
import anndata as ad

from .base import Task
from ..eval.metrics import pearson_per_row, mse, sign_accuracy_lineage


class NextTimepointTask(Task):
    """Predict pseudobulk expression at the next observed age from the current age.

    The grouping key (default ``organoid_id``) defines what a "trajectory" is.
    For per-organoid time-courses (synthetic data, Paola's organoids) use
    ``organoid_id``. For HNOCA — where each ``bio_sample`` is usually a single
    snapshot — use ``protocol`` so adjacent timepoints are paired *within* a
    protocol (cohort-level pseudobulk trajectory).
    """
    name = "t1_next_timepoint"

    def __init__(self, group_key: str = "organoid_id", holdout_organoid_frac: float = 0.2,
                 use_log1p: bool = True, hvg_top: int | None = None,
                 random_state: int = 0):
        self.group_key = group_key
        self.holdout_organoid_frac = holdout_organoid_frac
        self.use_log1p = use_log1p
        self.hvg_top = hvg_top
        self.random_state = random_state
        self.last_pairs_eval = None  # for plotting

    def _pseudobulk(self, adata: ad.AnnData) -> pd.DataFrame:
        df = pd.DataFrame(
            adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        df["age_days"] = adata.obs["age_days"].to_numpy()
        df[self.group_key] = adata.obs[self.group_key].astype(str).to_numpy()
        agg = df.groupby([self.group_key, "age_days"], observed=True).mean(numeric_only=True)
        if self.use_log1p:
            agg = np.log1p(agg)
        return agg.reset_index()

    def _make_pairs(self, pseudobulk: pd.DataFrame) -> pd.DataFrame:
        # Within each group (organoid or protocol), pair adjacent timepoints.
        rows = []
        for oid, sub in pseudobulk.groupby(self.group_key, observed=True):
            sub = sub.sort_values("age_days").reset_index(drop=True)
            for i in range(len(sub) - 1):
                rows.append({
                    self.group_key: oid,
                    "t": sub.loc[i, "age_days"],
                    "t_next": sub.loc[i + 1, "age_days"],
                    "x_t": sub.iloc[i].drop([self.group_key, "age_days"]).to_numpy(dtype=float),
                    "x_next": sub.iloc[i + 1].drop([self.group_key, "age_days"]).to_numpy(dtype=float),
                })
        return pd.DataFrame(rows)

    def prepare(self, adata: ad.AnnData) -> Tuple:
        if self.hvg_top is not None and self.hvg_top < adata.n_vars:
            # crude HVG: pick top-variance genes
            X = adata.X
            X = X.toarray() if hasattr(X, "toarray") else X
            var = X.var(axis=0)
            keep = np.argsort(-var)[: self.hvg_top]
            adata = adata[:, keep].copy()

        pb = self._pseudobulk(adata)
        pairs = self._make_pairs(pb)
        if len(pairs) == 0:
            empty = np.zeros((0, adata.n_vars))
            return empty, empty, empty, empty
        rng = np.random.default_rng(self.random_state)
        groups = pairs[self.group_key].unique()
        rng.shuffle(groups)
        n_eval = max(1, int(len(groups) * self.holdout_organoid_frac))
        eval_set = set(groups[:n_eval])
        train = pairs[~pairs[self.group_key].isin(eval_set)]
        evalp = pairs[pairs[self.group_key].isin(eval_set)]

        X_tr = np.stack(train["x_t"].to_list()) if len(train) else np.zeros((0, adata.n_vars))
        y_tr = np.stack(train["x_next"].to_list()) if len(train) else np.zeros((0, adata.n_vars))
        X_ev = np.stack(evalp["x_t"].to_list()) if len(evalp) else np.zeros((0, adata.n_vars))
        y_ev = np.stack(evalp["x_next"].to_list()) if len(evalp) else np.zeros((0, adata.n_vars))

        self.last_pairs_eval = evalp.reset_index(drop=True)
        return X_tr, y_tr, X_ev, y_ev

    def score(self, y_pred: np.ndarray, y_eval: np.ndarray):
        pearson = pearson_per_row(y_pred, y_eval).mean()
        mse_val = mse(y_pred, y_eval)
        # If we have a lineage mask handy via convention: first 50 cols ≡ lineage in synthetic;
        # in real HNOCA, plug in a curated TF list from configs/.
        n_lin = min(50, y_eval.shape[1])
        sign_acc = sign_accuracy_lineage(y_pred[:, :n_lin], y_eval[:, :n_lin])
        return {"pearson": float(pearson), "mse": float(mse_val), "sign_acc_lineage": float(sign_acc)}
