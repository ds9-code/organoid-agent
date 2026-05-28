"""T10 — Cross-Protocol Composition Transfer.

Given the cell-type composition of a *source* protocol at age T, predict the
composition of a *target* protocol at the same age T.

Why this is useful
------------------
Velasco at D90 ≠ Lancaster at D90. The interesting question is: how predictable
is the protocol-shift? If our agent can map Velasco -> Lancaster reliably, the
same machinery extends to "Velasco -> Paola's organoid at matched age" when
the Arlotta data lands.

Construction
------------
We pick the protocol pairs that have *matched-age* cells in HNOCA's Dorsal-
telencephalon subset. The cleanest pair is **Velasco ↔ Lancaster**, but we
also include any other protocols with mutual age coverage if available.

For each (source_protocol, target_protocol, age_bin) tuple, we compute the
source and target compositions, then evaluate predictors.

Predictors
----------
    identity      : predicted_target_composition = source_composition
    target_mean   : predicted = mean over all eval target compositions at any age
                    (chance baseline that ignores the source entirely)
    age_matched_target_mean : predicted = mean of OTHER training tuples'
                              targets at the same age_bin (closest realistic
                              non-cellflow baseline)

Metric
------
    KL(predicted || actual_target) on the proportion vector. Lower is better.

Run:
    PYTHONPATH=. python -m benchmarks.tasks.task_10_cross_protocol_transfer
"""
from __future__ import annotations

import argparse
import itertools
import numpy as np
import pandas as pd

from .common import (
    TaskResult, load_hnoca, kl_proportions, js_proportions, proportions, write_results,
)

TASK_ID = "T10_cross_protocol_transfer"


def _binned_age(ages: pd.Series, n_bins: int = 3) -> pd.Series:
    return pd.qcut(ages, q=n_bins, labels=False, duplicates="drop")


def _build_tuples(
    adata, granularity: str = "coarse", n_bins: int = 3,
    min_cells_per_bin: int = 30,
) -> pd.DataFrame:
    """Build (source_protocol, target_protocol, age_bin) tuples where both
    protocols have ≥``min_cells_per_bin`` cells in that bin."""
    label_col = "annot_level_2" if granularity == "coarse" else "cell_type"
    vocab = sorted(adata.obs[label_col].astype(str).unique().tolist())
    obs = adata.obs.copy()
    obs["age_bin"] = _binned_age(obs["age_days"], n_bins=n_bins)
    # Compositions per (protocol, age_bin)
    comp = {}
    for (proto, ab), sub in obs.groupby(["protocol_short", "age_bin"]):
        if len(sub) < min_cells_per_bin or pd.isna(ab):
            continue
        comp[(str(proto), int(ab))] = proportions(
            sub[label_col].astype(str).to_numpy(), vocab
        )
    rows = []
    for (p_s, b_s), c_s in comp.items():
        for (p_t, b_t), c_t in comp.items():
            if p_s == p_t or b_s != b_t:
                continue
            rows.append({
                "source_protocol": p_s,
                "target_protocol": p_t,
                "age_bin": b_s,
                "source_comp": c_s,
                "target_comp": c_t,
            })
    df = pd.DataFrame(rows)
    df.attrs["vocab"] = vocab
    return df


def _score(name: str, preds: list[np.ndarray], gts: list[np.ndarray]) -> TaskResult:
    kls = [kl_proportions(p, q) for p, q in zip(preds, gts)]
    jss = [js_proportions(p, q) for p, q in zip(preds, gts)]
    return TaskResult(
        task_id=TASK_ID, predictor=name,
        metrics={
            "mean_kl": float(np.mean(kls)),
            "median_kl": float(np.median(kls)),
            "mean_js": float(np.mean(jss)),
            "n_tuples": float(len(kls)),
        },
    )


def run(granularity: str = "coarse", n_bins: int = 3, holdout_target: str = "Lancaster",
        seed: int = 0) -> list[TaskResult]:
    print(f"[T10] loading HNOCA ...")
    adata = load_hnoca()
    tuples_df = _build_tuples(adata, granularity=granularity, n_bins=n_bins)
    if len(tuples_df) == 0:
        print("    no valid cross-protocol matched-age tuples; nothing to score")
        return []
    print(f"     {len(tuples_df)} (source, target, age_bin) tuples across "
          f"{tuples_df['source_protocol'].nunique()} unique source protocols, "
          f"{tuples_df['target_protocol'].nunique()} unique target protocols")

    # Hold out tuples where target_protocol == holdout_target — that's what we test
    eval_mask = tuples_df["target_protocol"] == holdout_target
    eval_df = tuples_df[eval_mask].reset_index(drop=True)
    train_df = tuples_df[~eval_mask].reset_index(drop=True)
    print(f"     holding out target_protocol='{holdout_target}': "
          f"{len(eval_df)} eval tuples, {len(train_df)} train tuples")

    if len(eval_df) == 0:
        print(f"    no tuples with target_protocol={holdout_target!r}; pick another")
        return []

    gts = [r["target_comp"] for _, r in eval_df.iterrows()]
    rows: list[TaskResult] = []

    # identity baseline
    rows.append(_score(
        "identity",
        preds=[r["source_comp"] for _, r in eval_df.iterrows()],
        gts=gts,
    ))

    # target_mean: mean of training target_comps regardless of age_bin
    if len(train_df):
        global_target_mean = np.mean(np.stack(train_df["target_comp"].to_list()), axis=0)
    else:
        global_target_mean = np.full_like(gts[0], 1.0 / len(gts[0]))
    rows.append(_score(
        "target_mean",
        preds=[global_target_mean] * len(eval_df),
        gts=gts,
    ))

    # age_matched_target_mean: mean training target_comp at the SAME age_bin
    mean_by_bin = {}
    for b, sub in train_df.groupby("age_bin"):
        mean_by_bin[int(b)] = np.mean(np.stack(sub["target_comp"].to_list()), axis=0)
    preds = []
    for _, r in eval_df.iterrows():
        preds.append(mean_by_bin.get(int(r["age_bin"]), global_target_mean))
    rows.append(_score("age_matched_target_mean", preds=preds, gts=gts))

    for row in rows:
        row.extras = {
            "granularity": granularity,
            "n_bins": int(n_bins),
            "holdout_target": holdout_target,
            "n_train_tuples": int(len(train_df)),
            "n_eval_tuples": int(len(eval_df)),
            "seed": int(seed),
            "eval_tuples": [
                {"source": r.source_protocol, "target": r.target_protocol,
                 "age_bin": int(r.age_bin)}
                for r in eval_df.itertuples()
            ],
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["fine", "coarse"], default="coarse")
    ap.add_argument("--n_bins", type=int, default=3)
    ap.add_argument("--holdout_target", default="Lancaster",
                    help="Protocol to hold out as the target_protocol. Default: Lancaster.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = run(granularity=args.granularity, n_bins=args.n_bins,
               holdout_target=args.holdout_target, seed=args.seed)
    if rows:
        write_results(rows, TASK_ID + f"_{args.granularity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
