"""T3 — Cell-Type Composition Shift Over Time (within a protocol).

Given the cell-type composition of a protocol at age t, predict the
composition at age t+Δ. The cleanest within-protocol time-extrapolation
benchmark we can build from HNOCA.

Construction
------------
For each protocol with ≥2 binned ages, we form ordered pairs of
(age_bin_source, age_bin_target). The bins are quartiles of the protocol's
own age range so every protocol contributes whatever it can.

Predictors
----------
    identity  : predict target composition = source composition
                (the "your organoid doesn't change with age" baseline)
    pop_mean  : predict target composition = mean composition across all
                training pairs at the same target_bin
    knn       : kNN retrieval — find pairs in *other* protocols at the
                matching (source_bin, target_bin), take the mean of their
                target compositions

Metric
------
    KL(predicted || actual) on the composition vector over annot_level_2
    (coarse, 3-class). Lower = better.

Run:
    PYTHONPATH=. python -m benchmarks.tasks.task_03_composition_shift
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from .common import (
    TaskResult, load_hnoca, kl_proportions, js_proportions, proportions,
    write_results,
)

TASK_ID = "T03_composition_shift"


def _binned_age(ages: pd.Series, n_bins: int = 4) -> pd.Series:
    """Quantile-bin ages into ``n_bins`` ordered bins per protocol."""
    return pd.qcut(ages, q=n_bins, labels=False, duplicates="drop")


def _build_pairs(adata, granularity: str = "coarse", n_bins: int = 4) -> pd.DataFrame:
    """For each protocol, build (source_bin, target_bin) pseudo-pairs of compositions."""
    label_col = "annot_level_2" if granularity == "coarse" else "cell_type"
    vocab = sorted(adata.obs[label_col].astype(str).unique().tolist())
    rows = []
    obs = adata.obs.copy()
    for proto, sub in obs.groupby("protocol_short"):
        if sub["age_days"].nunique() < 2:
            continue
        sub = sub.copy()
        sub["age_bin"] = _binned_age(sub["age_days"], n_bins=n_bins)
        bins = sorted(sub["age_bin"].dropna().unique().tolist())
        comp_by_bin = {}
        ages_by_bin = {}
        for b in bins:
            cells = sub[sub["age_bin"] == b]
            if len(cells) < 10:  # need at least a few cells to get stable composition
                continue
            comp_by_bin[b] = proportions(cells[label_col].astype(str).to_numpy(), vocab)
            ages_by_bin[b] = (float(cells["age_days"].min()), float(cells["age_days"].max()))
        for i, b in enumerate(bins[:-1]):
            for b_target in bins[i + 1:]:
                if b not in comp_by_bin or b_target not in comp_by_bin:
                    continue
                rows.append({
                    "protocol": str(proto),
                    "source_bin": int(b),
                    "target_bin": int(b_target),
                    "source_age_range": ages_by_bin[b],
                    "target_age_range": ages_by_bin[b_target],
                    "x_source": comp_by_bin[b],
                    "y_target": comp_by_bin[b_target],
                })
    df = pd.DataFrame(rows)
    df.attrs["vocab"] = vocab
    return df


def _score_predictor(name: str, preds: list[np.ndarray], gts: list[np.ndarray]) -> TaskResult:
    kls = [kl_proportions(p, q) for p, q in zip(preds, gts)]
    jss = [js_proportions(p, q) for p, q in zip(preds, gts)]
    return TaskResult(
        task_id=TASK_ID, predictor=name,
        metrics={
            "mean_kl": float(np.mean(kls)),
            "median_kl": float(np.median(kls)),
            "mean_js": float(np.mean(jss)),
            "n_pairs": float(len(kls)),
        },
    )


def run(granularity: str = "coarse", n_bins: int = 4, holdout_frac: float = 0.3,
        seed: int = 0) -> list[TaskResult]:
    print(f"[T3] loading HNOCA ...")
    adata = load_hnoca()
    pairs = _build_pairs(adata, granularity=granularity, n_bins=n_bins)
    if len(pairs) < 6:
        print(f"     only {len(pairs)} pairs available; results will be noisy")
    print(f"     {len(pairs)} (source_bin, target_bin) pairs across "
          f"{pairs['protocol'].nunique()} protocols")

    # Train/eval split by protocol — so the "knn" predictor genuinely has to
    # generalise from one protocol's trajectory to another
    rng = np.random.default_rng(seed)
    protocols = pairs["protocol"].unique().tolist()
    rng.shuffle(protocols)
    n_hold = max(1, int(len(protocols) * holdout_frac))
    eval_protos = set(protocols[:n_hold])
    train_pairs = pairs[~pairs["protocol"].isin(eval_protos)].reset_index(drop=True)
    eval_pairs = pairs[pairs["protocol"].isin(eval_protos)].reset_index(drop=True)
    print(f"     train pairs={len(train_pairs)} (protocols={[p for p in protocols if p not in eval_protos]})")
    print(f"     eval  pairs={len(eval_pairs)}  (protocols={list(eval_protos)})")

    gts = [r["y_target"] for _, r in eval_pairs.iterrows()]
    rows: list[TaskResult] = []

    # identity
    rows.append(_score_predictor(
        "identity",
        preds=[r["x_source"] for _, r in eval_pairs.iterrows()],
        gts=gts,
    ))

    # pop_mean: training mean composition at the matching target_bin
    pop_by_bin = {}
    for tgt_bin, sub in train_pairs.groupby("target_bin"):
        pop_by_bin[int(tgt_bin)] = np.mean(np.stack(sub["y_target"].to_list()), axis=0)
    global_pop = np.mean(np.stack(train_pairs["y_target"].to_list()), axis=0) if len(train_pairs) else None
    pop_preds = []
    for _, r in eval_pairs.iterrows():
        pop_preds.append(pop_by_bin.get(int(r["target_bin"]), global_pop))
    rows.append(_score_predictor("pop_mean", preds=pop_preds, gts=gts))

    # knn: for each eval pair, average y_target of train pairs with the same (source_bin, target_bin)
    knn_preds = []
    for _, r in eval_pairs.iterrows():
        match = train_pairs[
            (train_pairs["source_bin"] == r["source_bin"])
            & (train_pairs["target_bin"] == r["target_bin"])
        ]
        if len(match):
            knn_preds.append(np.mean(np.stack(match["y_target"].to_list()), axis=0))
        else:
            knn_preds.append(global_pop)
    rows.append(_score_predictor("knn_by_bins", preds=knn_preds, gts=gts))

    for row in rows:
        row.extras = {
            "granularity": granularity,
            "n_bins": int(n_bins),
            "holdout_frac": float(holdout_frac),
            "n_train_pairs": int(len(train_pairs)),
            "n_eval_pairs": int(len(eval_pairs)),
            "vocab": list(pairs.attrs["vocab"]),
            "seed": int(seed),
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["fine", "coarse"], default="coarse")
    ap.add_argument("--n_bins", type=int, default=4)
    ap.add_argument("--holdout_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = run(granularity=args.granularity, n_bins=args.n_bins,
               holdout_frac=args.holdout_frac, seed=args.seed)
    write_results(rows, TASK_ID + f"_{args.granularity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
