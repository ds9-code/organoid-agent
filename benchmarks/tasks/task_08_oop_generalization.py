"""T8 — Out-of-Protocol Cell-Type Generalisation.

Same classification problem as T2, but the train/test split is by **protocol**
instead of random cells. Train on the 4 largest protocols, test on the rest.
This is what tells us whether a classifier trained on Velasco organoids
generalises to a Bhaduri organoid it has never seen.

The headline number is **drop_in_macro_f1** = T2 macro-F1 minus T8 macro-F1.
A small drop means real generalisation; a big drop means the classifier
learned protocol-specific quirks.

Run:
    PYTHONPATH=. python -m benchmarks.tasks.task_08_oop_generalization
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from .common import (
    TaskResult, load_hnoca, log1p_pca_embed, protocol_split, write_results,
)

TASK_ID = "T08_oop_generalization"


def _score(name: str, y_pred, y_true) -> TaskResult:
    return TaskResult(
        task_id=TASK_ID, predictor=name,
        metrics={
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        },
    )


def run(
    granularity: str = "coarse",
    n_train_protocols: int = 4,
    seed: int = 0,
) -> list[TaskResult]:
    label_col = "annot_level_2" if granularity == "coarse" else "cell_type"

    print(f"[T8] loading HNOCA ...")
    adata = load_hnoca()
    counts = adata.obs["protocol_short"].value_counts()
    train_protocols = counts.head(n_train_protocols).index.tolist()
    eval_protocols = [p for p in counts.index if p not in train_protocols]
    print(f"     {adata.obs['protocol_short'].nunique()} protocols total")
    print(f"     train ({n_train_protocols}, largest):", train_protocols)
    print(f"     eval  ({len(eval_protocols)}):", eval_protocols[:6], "..." if len(eval_protocols) > 6 else "")

    # Split: TRAIN cells = cells from train protocols; EVAL cells = the rest.
    train_idx, eval_idx = protocol_split(adata, holdout_protocols=eval_protocols)
    y_train = adata.obs[label_col].astype(str).to_numpy()[train_idx]
    y_eval = adata.obs[label_col].astype(str).to_numpy()[eval_idx]
    print(f"     train={len(train_idx):,}  eval={len(eval_idx):,}")

    # Drop eval classes that were unseen in training (can't predict them)
    seen = set(np.unique(y_train))
    eval_keep = np.isin(y_eval, list(seen))
    if not eval_keep.all():
        n_drop = int((~eval_keep).sum())
        print(f"     dropping {n_drop} eval cells whose true label was unseen in train")
        eval_idx = eval_idx[eval_keep]
        y_eval = y_eval[eval_keep]

    # PCA fit ONLY on train cells, transform eval with same basis
    print(f"[T8] PCA-30 fit on train cells only ...")
    Z_train, hvg_idx, pca = log1p_pca_embed(adata[train_idx])
    Z_eval, _, _ = log1p_pca_embed(adata[eval_idx], hvg_idx=hvg_idx, pca=pca)

    rows: list[TaskResult] = []

    # Majority baseline
    maj = str(pd.Series(y_train).value_counts().idxmax())
    y_maj = np.full_like(y_eval, fill_value=maj, dtype=object)
    rows.append(_score("majority", y_maj, y_eval))
    print(f"     majority class on train set: {maj}")

    print(f"[T8] kNN k=15 ...")
    knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1).fit(Z_train, y_train)
    rows.append(_score("knn", knn.predict(Z_eval), y_eval))

    print(f"[T8] logreg ...")
    lr = LogisticRegression(max_iter=400, n_jobs=-1, C=1.0).fit(Z_train, y_train)
    rows.append(_score("logreg", lr.predict(Z_eval), y_eval))

    for r in rows:
        r.extras = {
            "granularity": granularity,
            "n_train_protocols": int(n_train_protocols),
            "train_protocols": list(train_protocols),
            "n_eval_protocols": int(len(eval_protocols)),
            "n_train_cells": int(len(train_idx)),
            "n_eval_cells": int(len(eval_idx)),
            "seed": int(seed),
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["fine", "coarse"], default="coarse")
    ap.add_argument("--n_train_protocols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = run(granularity=args.granularity, n_train_protocols=args.n_train_protocols, seed=args.seed)
    write_results(rows, TASK_ID + f"_{args.granularity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
