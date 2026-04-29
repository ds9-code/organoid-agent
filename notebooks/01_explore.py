"""Exploration script: builds synthetic HNOCA-shape data, generates plots,
runs T1/T2/T5 baselines, and writes a leaderboard PNG.

Run from the repo root:

    python notebooks/01_explore.py
"""
from __future__ import annotations
import os
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import make_synthetic_hnoca
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


def main():
    out = Path("plots")
    out.mkdir(exist_ok=True)
    adata = make_synthetic_hnoca(n_cells=8000, n_genes=400, n_organoids=30, seed=1)
    print(f"adata: {adata.shape}, regions={adata.obs['region'].nunique()}, "
          f"protocols={adata.obs['protocol'].nunique()}, "
          f"timepoints={sorted(adata.obs['age_days'].unique())}")

    # ---------- exploratory plots ----------
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_umap(adata, color="cell_type", ax=ax,
              title="Synthetic HNOCA-shape — coloured by cell type")
    fig.tight_layout(); fig.savefig(out / "01_pca_celltype.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_umap(adata, color="region", ax=ax,
              title="Synthetic HNOCA-shape — coloured by region")
    fig.tight_layout(); fig.savefig(out / "02_pca_region.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_celltype_composition(adata, ax=ax)
    fig.tight_layout(); fig.savefig(out / "03_composition_over_time.png", dpi=140); plt.close(fig)

    # pick a "lineage" gene (first 50 are lineage in the synthetic generator)
    gene = adata.var_names[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_time_course(adata, gene=gene, group_col="region", ax=ax)
    fig.tight_layout(); fig.savefig(out / "04_lineage_gene_timecourse.png", dpi=140); plt.close(fig)

    # ---------- run tasks ----------
    pairs = [
        (NextTimepointTask(hvg_top=200, holdout_organoid_frac=0.25), IdentityNextTimepoint()),
        (NextTimepointTask(hvg_top=200, holdout_organoid_frac=0.25), PopulationMeanNextTimepoint()),
        (NextTimepointTask(hvg_top=200, holdout_organoid_frac=0.25), LinearNextTimepoint(alpha=1.0)),
        (CellTypeOOPTask(holdout_frac=0.25), LogRegCellType(max_iter=200)),
        (CellTypeOOPTask(holdout_frac=0.25), KNNCellType(k=15)),
        (PseudotimeTask(hvg_top=200), PC1Pseudotime()),
    ]
    results = run_pairs(adata, pairs)
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

    # T1 pred-vs-true scatter for the linear baseline
    task1 = NextTimepointTask(hvg_top=200, holdout_organoid_frac=0.25)
    Xtr, ytr, Xev, yev = task1.prepare(adata)
    lin = LinearNextTimepoint(alpha=1.0).fit(Xtr, ytr)
    yhat = lin.predict(Xev)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plot_pred_vs_true_scatter(yhat, yev, ax=ax, title="T1 Linear: pred vs true (held-out organoids)")
    fig.tight_layout(); fig.savefig(out / "09_T1_linear_scatter.png", dpi=140); plt.close(fig)

    summary = {"adata_shape": list(adata.shape), "results": df.to_dict(orient="records")}
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote plots and results to {out.resolve()}")


if __name__ == "__main__":
    main()
