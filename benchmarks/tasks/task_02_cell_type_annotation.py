"""T2 — Automated Cell Type Annotation from Expression Profiles.

The bottleneck task: given a cell's gene-expression vector, predict its
cell-type label. Every other task that needs to know "what kind of cell is
this?" depends on this one.

Evaluation
----------
Random 80/20 stratified split (by cell-type label). Train PCA + classifier
on 80%, evaluate on the held-out 20%. Compare:

    - majority    : always predict the most-frequent training label
                    (random-chance baseline; what F1 looks like with zero signal)
    - knn (k=15)  : k-nearest-neighbours over PCA-30 latent
    - logreg      : multinomial logistic regression over PCA-30 latent

Reports macro-F1 (primary), accuracy, top-3 accuracy at fine and coarse
granularity.

Run:
    PYTHONPATH=. python -m benchmarks.tasks.task_02_cell_type_annotation
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from .common import (
    TaskResult, load_hnoca, log1p_pca_embed, stratified_cell_split, write_results,
)

TASK_ID = "T02_cell_type_annotation"


def _eval_predictor(name: str, y_pred, y_true, y_proba=None, classes=None) -> TaskResult:
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None and classes is not None:
        try:
            metrics["top3_accuracy"] = float(
                top_k_accuracy_score(y_true, y_proba, k=3, labels=classes)
            )
        except Exception:
            metrics["top3_accuracy"] = float("nan")
    return TaskResult(task_id=TASK_ID, predictor=name, metrics=metrics)


def run(granularity: str = "coarse", eval_frac: float = 0.2, seed: int = 0) -> list[TaskResult]:
    label_col = "annot_level_2" if granularity == "coarse" else "cell_type"

    print(f"[T2] loading HNOCA ...")
    adata = load_hnoca()
    labels = adata.obs[label_col].astype(str).to_numpy()
    print(f"     {adata.n_obs:,} cells, {len(np.unique(labels))} {granularity} classes "
          f"(top: {sorted(set(labels), key=lambda c: -(labels == c).sum())[:5]})")

    print(f"[T2] stratified split (eval_frac={eval_frac}, seed={seed}) ...")
    train_idx, eval_idx = stratified_cell_split(labels, eval_frac=eval_frac, seed=seed)
    y_train = labels[train_idx]
    y_eval = labels[eval_idx]
    print(f"     train={len(train_idx):,}  eval={len(eval_idx):,}")

    # Embedding fit ONLY on train cells (no test leakage), then transform eval
    print(f"[T2] computing PCA-30 on train cells only ...")
    Z_train, hvg_idx, pca = log1p_pca_embed(adata[train_idx])
    Z_eval, _, _ = log1p_pca_embed(adata[eval_idx], hvg_idx=hvg_idx, pca=pca)

    rows: list[TaskResult] = []

    # ---- (1) Majority-class baseline
    majority = pd.unique_majority(y_train)
    y_pred_maj = np.full_like(y_eval, fill_value=majority, dtype=object)
    rows.append(_eval_predictor("majority", y_pred_maj, y_eval))
    print(f"     majority class: {majority}")

    # ---- (2) k-NN k=15 on PCA latent
    print(f"[T2] kNN (k=15) on PCA-30 ...")
    knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1).fit(Z_train, y_train)
    y_pred_knn = knn.predict(Z_eval)
    y_proba_knn = knn.predict_proba(Z_eval)
    rows.append(_eval_predictor("knn", y_pred_knn, y_eval, y_proba_knn, knn.classes_))

    # ---- (3) Multinomial logistic regression
    print(f"[T2] logreg on PCA-30 ...")
    lr = LogisticRegression(max_iter=400, n_jobs=-1, C=1.0).fit(Z_train, y_train)
    y_pred_lr = lr.predict(Z_eval)
    y_proba_lr = lr.predict_proba(Z_eval)
    rows.append(_eval_predictor("logreg", y_pred_lr, y_eval, y_proba_lr, lr.classes_))

    # ---- attach scope info
    for r in rows:
        r.extras = {
            "granularity": granularity,
            "n_classes": int(len(np.unique(labels))),
            "n_train": int(len(train_idx)),
            "n_eval": int(len(eval_idx)),
            "seed": int(seed),
        }
    return rows


# small helper that doesn't deserve its own file
import pandas as _pd  # noqa: E402
class pd:
    @staticmethod
    def unique_majority(arr: np.ndarray) -> str:
        return str(_pd.Series(arr).value_counts().idxmax())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["fine", "coarse"], default="coarse")
    ap.add_argument("--eval_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = run(granularity=args.granularity, eval_frac=args.eval_frac, seed=args.seed)
    write_results(rows, TASK_ID + f"_{args.granularity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
