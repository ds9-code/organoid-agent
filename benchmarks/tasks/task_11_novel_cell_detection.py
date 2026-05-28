"""T11 — Novel-Cell Detection.

For each query cell, predict whether it has a confident analogue in the
atlas ("in-atlas", 0) or should be flagged as **novel** (1).

Operationalised here by holding out a protocol and treating its cells as
"novel". The detector trains on the remaining cells, then scores both
in-distribution (held-out cells of *included* protocols) and out-of-
distribution (held-out protocol) cells. AUROC on the binary label is the
headline metric — target ≥ 0.85 per our project notes.

This is the closest task to the eventual Arlotta use case: when Paola hands
us a new organoid, we want a calibrated "does this cell type live in HNOCA?"
score per cell.

Run:
    PYTHONPATH=. python -m benchmarks.tasks.task_11_novel_cell_detection
    PYTHONPATH=. python -m benchmarks.tasks.task_11_novel_cell_detection \\
        --novel_protocol Bhaduri
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score, roc_curve,
)
from sklearn.neighbors import NearestNeighbors

from .common import (
    TaskResult, load_hnoca, log1p_pca_embed, write_results,
)

TASK_ID = "T11_novel_cell_detection"


def _knn_distance_score(Z_train: np.ndarray, Z_query: np.ndarray, k: int = 15) -> np.ndarray:
    """Mean distance to k nearest neighbours in the training latent. Higher = more novel."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Z_train)
    dists, _ = nn.kneighbors(Z_query)
    return dists.mean(axis=1)


def _summary(scores: np.ndarray, labels: np.ndarray) -> dict:
    """AUROC, F1 at FPR≤5%, average precision."""
    auroc = float(roc_auc_score(labels, scores))
    avg_prec = float(average_precision_score(labels, scores))
    # F1 at FPR≤5%: find threshold s.t. FPR ≤ 0.05, return F1 at that op-point
    fpr, tpr, thr = roc_curve(labels, scores)
    op = np.argmin(np.abs(fpr - 0.05))
    op_thr = float(thr[op])
    pred = (scores >= op_thr).astype(int)
    f1_op = float(f1_score(labels, pred, zero_division=0))
    return {
        "auroc": auroc,
        "average_precision": avg_prec,
        "f1_at_fpr_5": f1_op,
        "threshold_at_fpr_5": op_thr,
    }


def run(
    novel_protocol: str | None = None,
    k: int = 15,
    seed: int = 0,
) -> list[TaskResult]:
    print(f"[T11] loading HNOCA ...")
    adata = load_hnoca()
    counts = adata.obs["protocol_short"].value_counts()

    # default: pick a "novel" protocol that has enough cells to evaluate cleanly
    # but is *not* the dominant one (otherwise we tear out half the atlas)
    if novel_protocol is None:
        # pick the largest protocol whose share is between 5%-20% of the subset
        for p, n in counts.items():
            frac = n / adata.n_obs
            if 0.03 <= frac <= 0.20:
                novel_protocol = p
                break
        if novel_protocol is None:
            novel_protocol = counts.index[1]  # second-largest as fallback
    n_novel = int(counts[novel_protocol])
    print(f"     designating {novel_protocol!r} as 'novel' ({n_novel} cells, "
          f"{n_novel / adata.n_obs:.1%} of subset)")

    is_novel = (adata.obs["protocol_short"] == novel_protocol).to_numpy()
    train_idx = np.flatnonzero(~is_novel)
    novel_idx = np.flatnonzero(is_novel)

    # in-distribution eval: take 20% of non-novel cells out to mix with novel
    rng = np.random.default_rng(seed)
    rng.shuffle(train_idx)
    n_in_eval = max(len(novel_idx), int(0.2 * len(train_idx)))
    in_eval_idx = train_idx[:n_in_eval]
    fit_idx = train_idx[n_in_eval:]
    print(f"     fit cells (in-distribution training):  {len(fit_idx):,}")
    print(f"     in-distribution eval cells:            {len(in_eval_idx):,}")
    print(f"     novel eval cells (held-out protocol):  {len(novel_idx):,}")

    # PCA fit on the in-distribution training cells only
    print(f"[T11] PCA-30 fit on in-distribution cells ...")
    Z_fit, hvg_idx, pca = log1p_pca_embed(adata[fit_idx])
    Z_in_eval, _, _ = log1p_pca_embed(adata[in_eval_idx], hvg_idx=hvg_idx, pca=pca)
    Z_novel, _, _ = log1p_pca_embed(adata[novel_idx], hvg_idx=hvg_idx, pca=pca)

    # Combined eval: novel cells labelled 1, in-distribution cells labelled 0
    Z_eval = np.vstack([Z_in_eval, Z_novel])
    labels = np.concatenate([np.zeros(len(Z_in_eval), dtype=int),
                              np.ones(len(Z_novel), dtype=int)])

    rows: list[TaskResult] = []

    # Random baseline (just to anchor what "no signal" looks like)
    rng2 = np.random.default_rng(seed)
    rand_scores = rng2.random(len(labels))
    rows.append(TaskResult(task_id=TASK_ID, predictor="random",
                            metrics=_summary(rand_scores, labels)))

    # kNN-distance: the simplest real novelty detector
    print(f"[T11] kNN-distance (k={k}) ...")
    knn_scores = _knn_distance_score(Z_fit, Z_eval, k=k)
    rows.append(TaskResult(task_id=TASK_ID, predictor=f"knn_distance_k{k}",
                            metrics=_summary(knn_scores, labels)))

    # kNN-distance with k=5 (less smoothing — typically a touch higher AUROC)
    print(f"[T11] kNN-distance (k=5) ...")
    knn5_scores = _knn_distance_score(Z_fit, Z_eval, k=5)
    rows.append(TaskResult(task_id=TASK_ID, predictor="knn_distance_k5",
                            metrics=_summary(knn5_scores, labels)))

    for r in rows:
        r.extras = {
            "novel_protocol": novel_protocol,
            "n_fit_cells": int(len(fit_idx)),
            "n_in_eval": int(len(Z_in_eval)),
            "n_novel_eval": int(len(Z_novel)),
            "seed": int(seed),
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--novel_protocol", default=None,
                    help="Protocol to designate as 'novel'. Default: auto-pick a "
                         "minor-protocol with 3-20%% of cells.")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = run(novel_protocol=args.novel_protocol, k=args.k, seed=args.seed)
    write_results(rows, TASK_ID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
