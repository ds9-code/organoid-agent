"""Plotting helpers — light wrappers over matplotlib so notebooks stay tidy."""
from __future__ import annotations
from typing import Dict, Iterable
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_umap(adata, color: str, ax=None, n_components: int = 2, title: str = ""):
    """Cheap 'UMAP-like' scatter using PCA when umap isn't installed.
    Replace with `sc.tl.umap` in a real environment."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X = np.log1p(X.astype(np.float32))
    Z = PCA(n_components=n_components, random_state=0).fit_transform(X)
    labels = adata.obs[color].astype(str).to_numpy()
    cats = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
    for i, c in enumerate(cats):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.6, label=c, color=cmap(i % cmap.N))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or f"PCA coloured by {color}")
    ax.legend(fontsize=7, markerscale=2, loc="best")
    return ax


def plot_time_course(adata, gene: str, group_col: str = "region", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    X = adata[:, gene].X
    X = X.toarray() if hasattr(X, "toarray") else X
    df = pd.DataFrame({
        "expr": np.log1p(X.ravel().astype(float)),
        "age_days": adata.obs["age_days"].to_numpy(),
        "group": adata.obs[group_col].astype(str).to_numpy(),
    })
    for grp, sub in df.groupby("group"):
        m = sub.groupby("age_days")["expr"].mean()
        ax.plot(m.index, m.values, marker="o", label=str(grp))
    ax.set_xlabel("Age (days)")
    ax.set_ylabel(f"log1p {gene}")
    ax.set_title(f"Time course of {gene} by {group_col}")
    ax.legend(fontsize=8)
    return ax


def plot_celltype_composition(adata, time_col: str = "age_days",
                              celltype_col: str = "cell_type", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    df = (adata.obs.groupby([time_col, celltype_col], observed=True)
                  .size().unstack(fill_value=0))
    df = df.div(df.sum(axis=1), axis=0)
    df.plot.area(ax=ax, alpha=0.85, colormap="tab20")
    ax.set_xlabel("Age (days)")
    ax.set_ylabel("Fraction of cells")
    ax.set_title("Cell-type composition over time")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return ax


def plot_task_leaderboard(scores: Dict[str, Dict[str, float]], metric: str, ax=None,
                          higher_better: bool = True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    rows = []
    for model_name, m in scores.items():
        if metric in m:
            rows.append((model_name, m[metric]))
    rows.sort(key=lambda r: r[1], reverse=higher_better)
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ax.barh(names[::-1], vals[::-1])
    ax.set_xlabel(metric + (" (↑)" if higher_better else " (↓)"))
    ax.set_title(f"Leaderboard — {metric}")
    return ax


def plot_pred_vs_true_scatter(y_pred, y_true, ax=None, title: str = "pred vs true"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    yp = np.ravel(y_pred)
    yt = np.ravel(y_true)
    ax.scatter(yt, yp, s=2, alpha=0.4)
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    return ax
