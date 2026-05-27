"""
Wrapper around the HNOCA subset used as a *tool* by the agent.

Expensive state -- the AnnData, log1p matrix, PCA basis, protocol/cell-type
indices -- is loaded once when ``HNOCAModel`` is constructed.  After that the
``predict_*`` / ``query_*`` methods answer specific organoid biology questions
*from real HNOCA cells*, so the LLM in :mod:`agent.agent` never invents numbers.

This mirrors the role of ``cellflow_model.py`` in mims-harvard/cellflow, but for
HNOCA-style benchmark questions rather than a trained perturbation model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # organoid_agent/tool_space -> repo root
DEFAULT_DATA = PROJECT_ROOT / "data" / "hnoca_dt_subset.h5ad"


def _short_protocol(name: str) -> str:
    """First-author short name for a HNOCA protocol string."""
    return str(name).split(",")[0].strip()


class HNOCAModel:
    """Load the real HNOCA subset and expose it as a small set of agent tools."""

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA,
        n_pcs: int = 30,
        verbose: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self._log = (lambda *a, **k: print(*a, **k)) if verbose else (lambda *a, **k: None)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"No HNOCA subset at {self.data_path}.\n"
                "Get it with: python download_data.py"
            )

        self._log(f"[1/3] Loading HNOCA subset from {self.data_path} ...")
        self.adata = ad.read_h5ad(self.data_path)
        self.adata.obs["protocol_short"] = (
            self.adata.obs["protocol"].astype(str).map(_short_protocol)
        )
        self._log(f"      {self.adata.n_obs:,} cells x {self.adata.n_vars:,} genes")

        self._log("[2/3] log1p-normalising + computing PCA basis...")
        X = self.adata.X
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        self._log1p = np.log1p(X_dense.astype(np.float32))
        var = self._log1p.var(axis=0)
        self._hvg_idx = np.argsort(-var)[:2000]
        self._pca = PCA(n_components=n_pcs, random_state=0).fit(self._log1p[:, self._hvg_idx])
        self._embed = self._pca.transform(self._log1p[:, self._hvg_idx])

        self._log("[3/4] Building kNN index over latent space...")
        self._knn = NearestNeighbors(n_neighbors=25, metric="euclidean").fit(self._embed)

        self._log("[4/4] Training cell-type classifiers (logreg + kNN) on HNOCA labels...")
        # fine labels
        self._fine_labels = self.adata.obs["cell_type"].astype(str).to_numpy()
        # coarse labels (annot_level_2 if available, else fine)
        coarse_col = "annot_level_2" if "annot_level_2" in self.adata.obs.columns else "cell_type"
        self._coarse_labels = self.adata.obs[coarse_col].astype(str).to_numpy()
        self._logreg_fine = LogisticRegression(max_iter=400, n_jobs=-1, C=1.0).fit(
            self._embed, self._fine_labels
        )
        self._logreg_coarse = LogisticRegression(max_iter=400, n_jobs=-1, C=1.0).fit(
            self._embed, self._coarse_labels
        )
        self._knn_clf = KNeighborsClassifier(n_neighbors=15, n_jobs=-1).fit(
            self._embed, self._fine_labels
        )

        self.protocols = sorted(self.adata.obs["protocol_short"].unique().tolist())
        self.cell_types = (
            self.adata.obs["cell_type"].value_counts().head(30).index.tolist()
        )
        self.age_min = int(self.adata.obs["age_days"].min())
        self.age_max = int(self.adata.obs["age_days"].max())
        self.gene_symbols = self.adata.var_names.tolist()
        self._log("      Ready.\n")

    # --------------------------------------------------------------- #
    # Introspection                                                   #
    # --------------------------------------------------------------- #
    def describe_inputs(self) -> dict[str, Any]:
        """Everything the LLM needs to construct a valid tool call."""
        baseline = (
            self.adata.obs["cell_type"]
            .value_counts(normalize=True)
            .head(10)
            .round(4)
            .to_dict()
        )
        return {
            "region": str(self.adata.uns.get("subset_region", "Dorsal telencephalon")),
            "n_cells": int(self.adata.n_obs),
            "n_genes": int(self.adata.n_vars),
            "n_protocols": len(self.protocols),
            "protocols": self.protocols,
            "age_days_range": [self.age_min, self.age_max],
            "available_cell_types_top30": [str(c) for c in self.cell_types],
            "baseline_cell_type_composition_top10": {str(k): float(v) for k, v in baseline.items()},
            "available_tools": [
                "query_composition",
                "gene_expression_timecourse",
                "predict_composition_at_age",
                "find_similar_cells",
                "classify_cell_type",
            ],
            "tool_input_notes": (
                "Use a protocol *short name* (first author, e.g. 'Velasco' / 'Lancaster' / "
                "'Pasca') -- the full DOIs are too long. Use exact GENE SYMBOLS like "
                "'NEUROD6', 'DCX', 'STMN2', 'VIM'. Ages are in days post-induction, "
                "see age_days_range. All answers come from real HNOCA cells in the "
                "Dorsal-telencephalon subset (~4.4k cells, 16 protocols, days 15-300)."
            ),
        }

    # --------------------------------------------------------------- #
    # Tools                                                           #
    # --------------------------------------------------------------- #
    def _filter_mask(
        self,
        protocol: str | None = None,
        age_min: float | None = None,
        age_max: float | None = None,
    ) -> np.ndarray:
        m = np.ones(self.adata.n_obs, dtype=bool)
        if protocol is not None:
            matches = self.adata.obs["protocol_short"] == protocol
            if not matches.any():
                raise ValueError(
                    f"Unknown protocol {protocol!r}. Available: {self.protocols}"
                )
            m &= matches.to_numpy()
        if age_min is not None:
            m &= (self.adata.obs["age_days"] >= float(age_min)).to_numpy()
        if age_max is not None:
            m &= (self.adata.obs["age_days"] <= float(age_max)).to_numpy()
        return m

    def query_composition(
        self,
        protocol: str | None = None,
        age_min: float | None = None,
        age_max: float | None = None,
    ) -> dict[str, Any]:
        """Cell-type composition of cells matching a filter (real HNOCA cells)."""
        m = self._filter_mask(protocol, age_min, age_max)
        n = int(m.sum())
        if n == 0:
            return {"n_cells": 0, "note": "No cells match filter; widen the criteria."}
        sub = self.adata.obs.loc[m]
        comp = sub["cell_type"].value_counts(normalize=True).round(4)
        coarse = (
            sub["annot_level_2"].value_counts(normalize=True).round(4).head(8).to_dict()
            if "annot_level_2" in sub.columns else {}
        )
        return {
            "filter": {"protocol": protocol, "age_min": age_min, "age_max": age_max},
            "n_cells": n,
            "n_organoids": int(sub["organoid_id"].nunique()),
            "n_protocols_in_filter": int(sub["protocol_short"].nunique()),
            "age_days_min_max": [float(sub["age_days"].min()), float(sub["age_days"].max())],
            "composition_top10": {str(k): float(v) for k, v in comp.head(10).items()},
            "coarse_composition_annot_level_2": {str(k): float(v) for k, v in coarse.items()},
        }

    def gene_expression_timecourse(
        self,
        gene: str,
        protocol: str | None = None,
    ) -> dict[str, Any]:
        """Mean log1p expression of one gene, by age bin (real HNOCA cells)."""
        if gene not in self.adata.var_names:
            # tiny fuzzy hint
            hits = [g for g in self.gene_symbols if gene.upper() in g.upper()][:8]
            raise ValueError(f"Gene {gene!r} not in atlas. Did you mean one of: {hits}?")
        m = self._filter_mask(protocol=protocol)
        if m.sum() == 0:
            return {"n_cells": 0, "note": "No cells match filter."}
        g_idx = self.adata.var_names.get_loc(gene)
        expr = self._log1p[m, g_idx]
        ages = self.adata.obs.loc[m, "age_days"].to_numpy()
        # bin ages into deciles of the subset's range for compactness
        bins = np.linspace(ages.min(), ages.max(), num=8 + 1)
        labels = []
        mean_expr = []
        n_per_bin = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (ages >= lo) & (ages <= hi)
            if sel.sum() < 3:
                continue
            labels.append(f"{int(lo)}-{int(hi)}d")
            mean_expr.append(round(float(expr[sel].mean()), 3))
            n_per_bin.append(int(sel.sum()))
        return {
            "gene": gene,
            "filter": {"protocol": protocol},
            "n_cells": int(m.sum()),
            "age_bins": labels,
            "mean_log1p_expression": mean_expr,
            "n_cells_per_bin": n_per_bin,
        }

    def predict_composition_at_age(
        self,
        protocol: str,
        target_age_days: float,
        k_neighbors: int = 50,
    ) -> dict[str, Any]:
        """Predict cell-type composition for a (protocol, age) by kNN retrieval
        in latent space from cells of that protocol at the *closest available*
        ages.  Useful proxy for "what should my Velasco-day-100 organoid look like?".
        """
        if protocol not in self.protocols:
            raise ValueError(f"Unknown protocol {protocol!r}. Available: {self.protocols}")
        m = self._filter_mask(protocol=protocol)
        if m.sum() == 0:
            return {"n_cells_seen": 0, "note": "No cells for this protocol in subset."}

        ages_avail = self.adata.obs.loc[m, "age_days"].to_numpy()
        # Take cells whose age is within +/- 10% of the target, falling back to k closest
        delta = np.abs(ages_avail - float(target_age_days))
        order = np.argsort(delta)
        sub_idx = np.flatnonzero(m)[order[:max(k_neighbors, 1)]]
        sub_ages = self.adata.obs["age_days"].iloc[sub_idx].to_numpy()

        comp = (
            self.adata.obs["cell_type"]
            .iloc[sub_idx]
            .value_counts(normalize=True)
            .round(4)
        )
        coarse = (
            self.adata.obs["annot_level_2"]
            .iloc[sub_idx]
            .value_counts(normalize=True)
            .round(4)
            .head(6)
            .to_dict()
            if "annot_level_2" in self.adata.obs.columns else {}
        )
        return {
            "protocol": protocol,
            "target_age_days": float(target_age_days),
            "method": "kNN retrieval over real HNOCA cells at nearest ages",
            "k_neighbors_used": int(len(sub_idx)),
            "actual_ages_used_min_max": [float(sub_ages.min()), float(sub_ages.max())],
            "ages_extrapolated": bool(
                float(target_age_days) < ages_avail.min()
                or float(target_age_days) > ages_avail.max()
            ),
            "predicted_composition_top10": {str(k): float(v) for k, v in comp.head(10).items()},
            "predicted_coarse_composition": {str(k): float(v) for k, v in coarse.items()},
        }

    def classify_cell_type(
        self,
        protocol: str,
        age_days: float,
        granularity: str = "coarse",
        model: str = "logreg",
        n_cells: int = 100,
    ) -> dict[str, Any]:
        """Classify cells of a (protocol, age) sample using a model trained on HNOCA.

        Returns the predicted label distribution + per-label confidence.

        Parameters
        ----------
        protocol, age_days : query cohort
        granularity : "fine" (28 cell_type classes) or "coarse" (annot_level_2)
        model : "logreg" (multinomial logistic regression) or "knn" (k=15)
        n_cells : how many real HNOCA cells from this cohort to classify

        The classifier was *trained* on HNOCA — so on in-distribution cohorts it
        should mostly reproduce the true labels. The point of exposing it as a
        tool is so that on *novel* cohorts (e.g. Paola's organoids) the same
        classifier can be applied and we know its training distribution.
        """
        if protocol not in self.protocols:
            raise ValueError(f"Unknown protocol {protocol!r}. Available: {self.protocols}")
        m = self._filter_mask(protocol=protocol, age_min=age_days * 0.9, age_max=age_days * 1.1)
        if m.sum() == 0:
            m = self._filter_mask(protocol=protocol)
            if m.sum() == 0:
                raise ValueError(f"No cells for protocol {protocol!r}")

        idx = np.flatnonzero(m)[:n_cells]
        X = self._embed[idx]
        clf = {
            "logreg": (self._logreg_coarse if granularity == "coarse" else self._logreg_fine),
            "knn": self._knn_clf,
        }.get(model)
        if clf is None:
            raise ValueError(f"Unknown model {model!r}. Options: 'logreg', 'knn'.")

        preds = clf.predict(X)
        try:
            probs = clf.predict_proba(X).mean(axis=0)
            classes = clf.classes_
            top = sorted(
                ((str(c), float(p)) for c, p in zip(classes, probs)),
                key=lambda kv: -kv[1],
            )[:8]
        except Exception:
            top = []

        true_labels = (
            self.adata.obs["annot_level_2"].iloc[idx].astype(str).to_numpy()
            if granularity == "coarse" and "annot_level_2" in self.adata.obs.columns
            else self.adata.obs["cell_type"].iloc[idx].astype(str).to_numpy()
        )
        # in-distribution self-check: did the classifier match the atlas labels?
        accuracy = float((preds == true_labels).mean())

        pred_props = (
            pd.Series(preds).value_counts(normalize=True).round(4).head(8).to_dict()
        )
        return {
            "query": {"protocol": protocol, "age_days": float(age_days),
                      "granularity": granularity, "model": model, "n_cells": int(len(idx))},
            "predicted_composition": {str(k): float(v) for k, v in pred_props.items()},
            "mean_class_probability_top8": [
                {"label": label, "prob": round(p, 4)} for label, p in top
            ],
            "in_distribution_accuracy_vs_atlas_labels": round(accuracy, 4),
            "note": (
                "Accuracy is measured against the atlas labels themselves -- "
                "training and eval cells overlap. Treat this as a sanity check that the "
                "classifier reproduces the labels it was trained on, not as a held-out score."
            ),
        }

    def find_similar_cells(
        self,
        protocol: str,
        age_days: float,
        k: int = 25,
    ) -> dict[str, Any]:
        """For a (protocol, age) query, return the dominant cell-type / protocol
        mix of its k nearest neighbours in 30-D PCA latent space.

        This is the "what does this experimental condition look like in the
        atlas?" tool -- the analogue of HNOCA-tools `map_query` for Paola's
        organoids, but using only HNOCA's own cells as a sanity probe.
        """
        m = self._filter_mask(protocol=protocol, age_min=age_days * 0.9, age_max=age_days * 1.1)
        if m.sum() == 0:
            m = self._filter_mask(protocol=protocol)
            if m.sum() == 0:
                raise ValueError(f"No cells for protocol {protocol!r}")
        query_centroid = self._embed[m].mean(axis=0, keepdims=True)
        dists, idx = self._knn.kneighbors(query_centroid, n_neighbors=min(k, self._embed.shape[0]))
        idx = idx.ravel()
        ngh = self.adata.obs.iloc[idx]
        return {
            "query": {"protocol": protocol, "age_days": float(age_days), "k": int(len(idx))},
            "neighbour_protocols": {
                str(k): int(v)
                for k, v in ngh["protocol_short"].value_counts().head(6).items()
            },
            "neighbour_cell_types": {
                str(k): int(v)
                for k, v in ngh["cell_type"].value_counts().head(8).items()
            },
            "neighbour_age_days_mean": float(ngh["age_days"].mean()),
            "mean_latent_distance": float(dists.mean()),
        }
