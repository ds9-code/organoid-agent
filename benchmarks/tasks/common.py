"""Shared utilities for per-task evaluations.

Light helpers used by every benchmark in this folder:
  - load_hnoca()                 : load the streamed HNOCA subset
  - log1p_pca_embed()            : log1p -> top-K HVGs -> PCA-30 (matches the
                                    same recipe used by HNOCAModel)
  - stratified_cell_split()      : random N/M split stratified by cell type
  - protocol_split()             : split by protocol (for T8 + T11)
  - kl_proportions(), js_proportions() : composition-vector distances
  - write_results(rows, task_id) : append-only CSV writer
"""
from __future__ import annotations

import json
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "hnoca_dt_subset.h5ad"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


def load_hnoca(path: str | Path = DEFAULT_DATA) -> ad.AnnData:
    """Load the streamed HNOCA subset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run: python download_data.py"
        )
    adata = ad.read_h5ad(p)
    # Normalize protocol_short the same way HNOCAModel does
    adata.obs["protocol_short"] = (
        adata.obs["protocol"].astype(str).str.split(",").str[0].str.strip()
    )
    return adata


def log1p_pca_embed(
    adata: ad.AnnData, n_hvg: int = 2000, n_pcs: int = 30, hvg_idx: np.ndarray | None = None,
    pca: PCA | None = None,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """log1p -> top-N HVGs -> PCA-30. Same recipe as HNOCAModel.

    If ``hvg_idx`` / ``pca`` are passed, reuse them (for test-set embedding
    using train-set basis). Otherwise fit fresh.
    """
    X = adata.X
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    L = np.log1p(X_dense.astype(np.float32))
    if hvg_idx is None:
        var = L.var(axis=0)
        hvg_idx = np.argsort(-var)[:n_hvg]
    if pca is None:
        pca = PCA(n_components=n_pcs, random_state=0).fit(L[:, hvg_idx])
    embed = pca.transform(L[:, hvg_idx])
    return embed, hvg_idx, pca


def stratified_cell_split(
    labels: np.ndarray, eval_frac: float = 0.2, seed: int = 0,
    min_per_class: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split by cell label.

    Drops classes with fewer than ``min_per_class`` cells (can't stratify).
    Returns (train_idx, eval_idx).
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    keep = set(classes[counts >= min_per_class])
    valid_idx = np.flatnonzero(np.isin(labels, list(keep)))

    train, evalp = [], []
    for c in keep:
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_eval = max(1, int(round(len(idx) * eval_frac)))
        evalp.extend(idx[:n_eval].tolist())
        train.extend(idx[n_eval:].tolist())
    return np.array(sorted(train)), np.array(sorted(evalp))


def protocol_split(
    adata: ad.AnnData, holdout_protocols: Iterable[str],
    protocol_col: str = "protocol_short",
) -> tuple[np.ndarray, np.ndarray]:
    """Split by ``protocol_short``. Returns (train_idx, eval_idx)."""
    hold = set(holdout_protocols)
    is_eval = adata.obs[protocol_col].astype(str).isin(hold).to_numpy()
    eval_idx = np.flatnonzero(is_eval)
    train_idx = np.flatnonzero(~is_eval)
    return train_idx, eval_idx


def kl_proportions(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q) on proportion vectors. Both sum to 1 by assumption."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float((p * np.log(p / q)).sum())


def js_proportions(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Jensen–Shannon divergence on proportion vectors. Symmetric in [0, ln 2]."""
    m = 0.5 * (np.asarray(p, dtype=float) + np.asarray(q, dtype=float))
    return 0.5 * kl_proportions(p, m, eps) + 0.5 * kl_proportions(q, m, eps)


def proportions(labels: np.ndarray, vocab: list[str]) -> np.ndarray:
    """Counts of `labels` over `vocab`, normalised to proportions."""
    s = pd.Series(labels).value_counts()
    return np.array([float(s.get(v, 0)) / max(len(labels), 1) for v in vocab])


# ------------------------------------------------------------- #
# Result I/O                                                    #
# ------------------------------------------------------------- #

@dataclass
class TaskResult:
    task_id: str
    predictor: str
    metrics: dict[str, float] = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


def write_results(rows: list[TaskResult], task_id: str) -> Path:
    """Write per-predictor metrics to ``benchmarks/results/<task>_<timestamp>.json``
    and print a small Markdown-style table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{task_id}_{ts}.json"
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2))

    # printed summary
    all_metrics = sorted({m for r in rows for m in r.metrics})
    header = ["predictor"] + all_metrics
    widths = [max(len(h), 12) for h in header]
    print()
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        cells = [r.predictor] + [f"{r.metrics.get(m, float('nan')):.4f}" for m in all_metrics]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
    print(f"\nwrote: {out.relative_to(PROJECT_ROOT)}")
    return out
