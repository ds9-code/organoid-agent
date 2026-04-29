"""T1 — Next-timepoint pseudobulk prediction."""
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
import anndata as ad

from .base import Task
from ..eval.metrics import pearson_per_row, mse, sign_accuracy_lineage


class NextTimepointTask(Task):
    name = "t1_next_timepoint"

    def __init__(self, group_keys=("organoid_id",), holdout_organoid_frac: float = 0.2,
                 use_log1p: bool = True, hvg_top: int | None = None,
                 random_state: int = 0):
        self.group_keys = group_keys
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
        for k in self.group_keys:
            df[k] = adata.obs[k].to_numpy()
        keys = list(self.group_keys) + ["age_days"]
        agg = df.groupby(keys, observed=True).mean(numeric_only=True)
        if self.use_log1p:
            agg = np.log1p(agg)
        return agg.reset_index()

    def _make_pairs(self, pseudobulk: pd.DataFrame) -> pd.DataFrame:
        # For each organoid, pair adjacent timepoints.
        rows = []
        for oid, sub in pseudobulk.groupby("organoid_id", observed=True):
            sub = sub.sort_values("age_days").reset_index(drop=True)
            for i in range(len(sub) - 1):
                rows.append({
                    "organoid_id": oid,
                    "t": sub.loc[i, "age_days"],
                    "t_next": sub.loc[i + 1, "age_days"],
                    "x_t": sub.iloc[i].drop(["organoid_id", "age_days"]).to_numpy(dtype=float),
                    "x_next": sub.iloc[i + 1].drop(["organoid_id", "age_days"]).to_numpy(dtype=float),
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
        rng = np.random.default_rng(self.random_state)
        organoids = pairs["organoid_id"].unique()
        rng.shuffle(organoids)
        n_eval = max(1, int(len(organoids) * self.holdout_organoid_frac))
        eval_set = set(organoids[:n_eval])
        train = pairs[~pairs["organoid_id"].isin(eval_set)]
        evalp = pairs[pairs["organoid_id"].isin(eval_set)]

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
