"""Exploration script: loads the real HNOCA subset, generates plots, runs
T1/T2/T5 baselines, and writes a leaderboard PNG.

Run from the repo root:

    python notebooks/01_explore.py

Requires data/hnoca_dt_subset.h5ad — get it with:

    python download_data.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import json

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.tasks import NextTimepointTask, CellTypeOOPTask, PseudotimeTask
from src.models import (
    IdentityNextTimepoint, PopulationMeanNextTimepoint, LinearNextTimepoint,
    LogRegCellType, KNNCellType, PC1Pseudotime,
)
from src.plot import (
    plot_umap, plot_time_course, plot_celltype_composition,
    plot_task_leaderboard, plot_pred_vs_true_scatter,
)
from src.eval.runner import run_pairs, to_long_table

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "hnoca_dt_subset.h5ad"


def main():
    out = Path("plots")
    out.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run: python download_data.py", file=sys.stderr)
        return 1
    print(f"Loading real HNOCA subset from {DATA_PATH} ...")
    adata = ad.read_h5ad(DATA_PATH)
    print(f"adata: {adata.shape}, region={adata.uns.get('subset_region','?')}, "
          f"protocols={adata.obs['protocol'].nunique()}, "
          f"n_organoids={adata.obs['organoid_id'].nunique()}, "
          f"timepoints={adata.obs['age_days'].nunique()} unique "
          f"(range {int(adata.obs['age_days'].min())}-{int(adata.obs['age_days'].max())} days)")

    # ---------- exploratory plots ----------
    # Collapse very rare cell types into "other" for legibility (atlas has many leaves)
    ct_counts = adata.obs["cell_type"].value_counts()
    keep_ct = set(ct_counts.head(10).index)
    ct_str = adata.obs["cell_type"].astype(str)
    adata.obs["cell_type_top"] = np.where(ct_str.isin(keep_ct), ct_str, "other")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    plot_umap(adata, color="cell_type_top", ax=ax,
              title="HNOCA Dorsal-telencephalon subset — cell type (top 10 + other)")
    fig.tight_layout(); fig.savefig(out / "01_pca_celltype.png", dpi=140); plt.close(fig)

    # Group protocol by first-author for legibility
    adata.obs["protocol_short"] = (
        adata.obs["protocol"].astype(str).str.split(",").str[0]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_umap(adata, color="protocol_short", ax=ax,
              title="HNOCA Dorsal-telencephalon subset — protocol")
    fig.tight_layout(); fig.savefig(out / "02_pca_protocol.png", dpi=140); plt.close(fig)

    # Composition over time — use coarser annot_level_2 if available, else cell_type_top
    comp_col = "annot_level_2" if "annot_level_2" in adata.obs.columns else "cell_type_top"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_celltype_composition(adata, celltype_col=comp_col, ax=ax)
    ax.set_title(f"HNOCA — composition over real organoid age ({comp_col})")
    fig.tight_layout(); fig.savefig(out / "03_composition_over_time.png", dpi=140); plt.close(fig)

    # Time course for a canonical cortical-neuron marker (NEUROD6) — fall back if missing
    for gene_candidate in ("NEUROD6", "DCX", "STMN2", "VIM", adata.var_names[0]):
        if gene_candidate in adata.var_names:
            gene = gene_candidate
            break
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plot_time_course(adata, gene=gene, group_col="protocol_short", ax=ax)
    ax.set_title(f"HNOCA — {gene} over organoid age, by protocol")
    fig.tight_layout(); fig.savefig(out / "04_marker_gene_timecourse.png", dpi=140); plt.close(fig)

    # ---------- run tasks on real data ----------
    # In HNOCA each `bio_sample`(=organoid_id) is usually a single snapshot, so we
    # pair adjacent timepoints WITHIN a protocol — a cohort-level pseudobulk
    # trajectory rather than per-organoid.
    n_tp_per_protocol = adata.obs.groupby("protocol", observed=True)["age_days"].nunique()
    multi_tp_protocols = n_tp_per_protocol[n_tp_per_protocol >= 2].index
    print(f"T1 setup: {len(multi_tp_protocols)} protocols have >=2 timepoints "
          f"(of {adata.obs['protocol'].nunique()} total)")
    t1_adata = adata[adata.obs["protocol"].isin(multi_tp_protocols)].copy() if len(multi_tp_protocols) else adata

    t1_kwargs = dict(group_key="protocol", hvg_top=500, holdout_organoid_frac=0.25)
    pairs = [
        (NextTimepointTask(**t1_kwargs), IdentityNextTimepoint()),
        (NextTimepointTask(**t1_kwargs), PopulationMeanNextTimepoint()),
        (NextTimepointTask(**t1_kwargs), LinearNextTimepoint(alpha=1.0)),
        (CellTypeOOPTask(holdout_frac=0.25), LogRegCellType(max_iter=300)),
        (CellTypeOOPTask(holdout_frac=0.25), KNNCellType(k=15)),
        (PseudotimeTask(hvg_top=500), PC1Pseudotime()),
    ]
    from src.eval.runner import run_pairs as _run
    results = []
    for task, model in pairs:
        ad_in = t1_adata if task.name == "t1_next_timepoint" else adata
        results.extend(_run(ad_in, [(task, model)]))
    df = to_long_table(results)
    print(df.to_string(index=False))
    df.to_csv(out / "results_long.csv", index=False)

    # ---------- leaderboards ----------
    # T1 by Pearson
    t1_scores = {r.extras["model"]: r.metrics for r in results if r.name == "t1_next_timepoint"}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_task_leaderboard(t1_scores, "pearson", ax=ax, higher_better=True)
    fig.tight_layout(); fig.savefig(out / "05_T1_pearson_leaderboard.png", dpi=140); plt.close(fig)

    # T1 by MSE (lower is better)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_task_leaderboard(t1_scores, "mse", ax=ax, higher_better=False)
    fig.tight_layout(); fig.savefig(out / "06_T1_mse_leaderboard.png", dpi=140); plt.close(fig)

    # T2 macro F1
    t2_scores = {r.extras["model"]: r.metrics for r in results if r.name == "t2_celltype_oop"}
    fig, ax = plt.subplots(figsize=(6, 3.0))
    plot_task_leaderboard(t2_scores, "macro_f1", ax=ax, higher_better=True)
    fig.tight_layout(); fig.savefig(out / "07_T2_macro_f1_leaderboard.png", dpi=140); plt.close(fig)

    # T5 spearman
    t5_scores = {r.extras["model"]: r.metrics for r in results if r.name == "t5_pseudotime"}
    fig, ax = plt.subplots(figsize=(6, 2.5))
    plot_task_leaderboard(t5_scores, "spearman", ax=ax, higher_better=True)
    fig.tight_layout(); fig.savefig(out / "08_T5_spearman_leaderboard.png", dpi=140); plt.close(fig)

    # T1 pred-vs-true scatter for the linear baseline (on real HNOCA pseudobulks)
    task1 = NextTimepointTask(**t1_kwargs)
    Xtr, ytr, Xev, yev = task1.prepare(t1_adata)
    if Xev.shape[0] > 0:
        lin = LinearNextTimepoint(alpha=1.0).fit(Xtr, ytr)
        yhat = lin.predict(Xev)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        plot_pred_vs_true_scatter(yhat, yev, ax=ax,
                                  title="HNOCA T1 Linear: pred vs true (held-out organoids)")
        fig.tight_layout(); fig.savefig(out / "09_T1_linear_scatter.png", dpi=140); plt.close(fig)
    else:
        print("Skipping T1 scatter — no held-out organoid pairs available.")

    summary = {
        "data_source": str(DATA_PATH),
        "adata_shape": list(adata.shape),
        "subset_region": adata.uns.get("subset_region"),
        "n_protocols": int(adata.obs["protocol"].nunique()),
        "n_organoids": int(adata.obs["organoid_id"].nunique()),
        "n_timepoints": int(adata.obs["age_days"].nunique()),
        "age_days_range": [int(adata.obs["age_days"].min()), int(adata.obs["age_days"].max())],
        "results": df.to_dict(orient="records"),
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote plots and results to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
