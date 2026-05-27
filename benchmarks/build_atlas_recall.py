"""
Build the atlas-recall question set by computing ground-truth answers from the
HNOCA subset directly. Writes ``benchmarks/questions/atlas_recall.yaml``.

Each question has:
  - id: short identifier
  - question: the natural-language string we'll ask the agent
  - expected_tool: which HNOCAModel method, if any, should be called
  - expected_answer: ground-truth value (scalar / dict / list)
  - tolerance: numeric tolerance (or null for categorical)
  - notes: free-text rationale

Run:  python benchmarks/build_atlas_recall.py
"""
from __future__ import annotations
from pathlib import Path
import json
import yaml
import numpy as np
import anndata as ad

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "hnoca_dt_subset.h5ad"
OUT_PATH = PROJECT_ROOT / "benchmarks" / "questions" / "atlas_recall.yaml"


def _short(p: str) -> str:
    return str(p).split(",")[0].strip()


def main() -> None:
    print(f"Loading {DATA_PATH} ...")
    adata = ad.read_h5ad(DATA_PATH)
    adata.obs["protocol_short"] = adata.obs["protocol"].astype(str).map(_short)
    obs = adata.obs

    # Pre-compute everything once so the YAML is deterministic.
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    log1p = np.log1p(X.astype(np.float32))

    def gene_idx(g):
        return adata.var_names.get_loc(g) if g in adata.var_names else None

    questions = []

    # ----- composition questions -----
    # A1: top-3 cell types in Velasco day 80-120
    m = ((obs["protocol_short"] == "Velasco")
         & (obs["age_days"] >= 80) & (obs["age_days"] <= 120)).to_numpy()
    top3 = (
        obs.loc[m, "cell_type"].value_counts(normalize=True).round(3).head(3).to_dict()
    )
    questions.append({
        "id": "A1",
        "question": "What are the top 3 cell types in Velasco-protocol organoids between day 80 and day 120, by fraction?",
        "expected_tool": "query_composition",
        "expected_answer": {str(k): float(v) for k, v in top3.items()},
        "tolerance": 0.05,
        "notes": f"n_cells in filter = {int(m.sum())}",
    })

    # A2: fraction of radial glia in subset
    frac_rg = float((obs["cell_type"] == "radial glial cell").mean().round(4))
    questions.append({
        "id": "A2",
        "question": "Across the entire HNOCA dorsal-telencephalon subset, what fraction of cells are labelled 'radial glial cell'?",
        "expected_tool": "query_composition",
        "expected_answer": frac_rg,
        "tolerance": 0.01,
        "notes": "",
    })

    # A3: number of unique cell-type labels + unknown fraction
    n_labels = int(obs["cell_type"].nunique())
    frac_unknown = float((obs["cell_type"] == "unknown").mean().round(4))
    questions.append({
        "id": "A3",
        "question": "How many distinct cell-type labels exist in the HNOCA dorsal-telencephalon subset, and what fraction of cells are labelled 'unknown'?",
        "expected_tool": "describe_inputs",
        "expected_answer": {"n_labels": n_labels, "frac_unknown": frac_unknown},
        "tolerance": 0.01,
        "notes": "",
    })

    # A4: dominant annot_level_2
    if "annot_level_2" in obs.columns:
        top = obs["annot_level_2"].value_counts(normalize=True).round(4).head(1)
        dominant_label = str(top.index[0])
        dominant_frac = float(top.iloc[0])
    else:
        dominant_label, dominant_frac = ("?", 0.0)
    questions.append({
        "id": "A4",
        "question": "What is the dominant coarse cell class (annot_level_2) in the HNOCA dorsal-telencephalon subset, and what fraction of cells does it cover?",
        "expected_tool": "query_composition",
        "expected_answer": {"label": dominant_label, "frac": dominant_frac},
        "tolerance": 0.02,
        "notes": "",
    })

    # ----- protocol questions -----
    # A5: number of unique protocols
    n_protocols = int(obs["protocol_short"].nunique())
    questions.append({
        "id": "A5",
        "question": "How many distinct organoid differentiation protocols are represented in the HNOCA dorsal-telencephalon subset?",
        "expected_tool": "describe_inputs",
        "expected_answer": n_protocols,
        "tolerance": 0,
        "notes": "",
    })

    # A6: largest-cell-count protocol
    biggest = obs["protocol_short"].value_counts().head(1)
    questions.append({
        "id": "A6",
        "question": "Which protocol contributes the most cells to the HNOCA dorsal-telencephalon subset, and how many cells does it contribute?",
        "expected_tool": "query_composition",
        "expected_answer": {"protocol": str(biggest.index[0]), "n_cells": int(biggest.iloc[0])},
        "tolerance": None,
        "notes": "categorical + integer",
    })

    # ----- age / time-course questions -----
    # A7: age range
    questions.append({
        "id": "A7",
        "question": "What is the range of organoid ages (in days) covered by the HNOCA dorsal-telencephalon subset?",
        "expected_tool": "describe_inputs",
        "expected_answer": {"min_days": int(obs["age_days"].min()), "max_days": int(obs["age_days"].max())},
        "tolerance": 0,
        "notes": "",
    })

    # A8: NEUROD6 timecourse peak bin
    g = gene_idx("NEUROD6")
    if g is not None:
        ages = obs["age_days"].to_numpy()
        expr = log1p[:, g]
        bins = np.linspace(ages.min(), ages.max(), 8 + 1)
        bin_means = []
        bin_labels = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (ages >= lo) & (ages <= hi)
            if sel.sum() < 3:
                continue
            bin_means.append(float(expr[sel].mean()))
            bin_labels.append(f"{int(lo)}-{int(hi)}d")
        peak_idx = int(np.argmax(bin_means))
        peak_label = bin_labels[peak_idx]
        peak_value = round(bin_means[peak_idx], 2)
    else:
        peak_label, peak_value = ("?", 0.0)
    questions.append({
        "id": "A8",
        "question": "Across the entire HNOCA dorsal-telencephalon subset, in which age range (days) does mean log1p expression of NEUROD6 peak?",
        "expected_tool": "gene_expression_timecourse",
        "expected_answer": {"peak_age_bin": peak_label, "peak_log1p_expression": peak_value},
        "tolerance": 0.2,
        "notes": "",
    })

    # A9: VIM vs DCX comparison day 30 vs day 150
    g_vim = gene_idx("VIM")
    g_dcx = gene_idx("DCX")
    if g_vim is not None and g_dcx is not None:
        m_30 = ((obs["age_days"] >= 25) & (obs["age_days"] <= 40)).to_numpy()
        m_150 = ((obs["age_days"] >= 140) & (obs["age_days"] <= 170)).to_numpy()
        vim_30 = float(log1p[m_30, g_vim].mean()) if m_30.sum() else 0.0
        vim_150 = float(log1p[m_150, g_vim].mean()) if m_150.sum() else 0.0
        dcx_30 = float(log1p[m_30, g_dcx].mean()) if m_30.sum() else 0.0
        dcx_150 = float(log1p[m_150, g_dcx].mean()) if m_150.sum() else 0.0
        answer = {
            "vim_decreases_with_age": bool(vim_150 < vim_30),
            "dcx_increases_with_age": bool(dcx_150 > dcx_30),
            "vim_d30": round(vim_30, 2),
            "vim_d150": round(vim_150, 2),
            "dcx_d30": round(dcx_30, 2),
            "dcx_d150": round(dcx_150, 2),
        }
    else:
        answer = {"note": "VIM/DCX not in subset"}
    questions.append({
        "id": "A9",
        "question": "In the HNOCA dorsal-telencephalon subset, does VIM (radial-glia marker) expression decrease and DCX (immature-neuron marker) expression increase between day 30 and day 150?",
        "expected_tool": "gene_expression_timecourse",
        "expected_answer": answer,
        "tolerance": None,
        "notes": "qualitative answers about direction of change",
    })

    # ----- per-organoid / structural questions -----
    # A10: organoids with >=2 timepoints (the Pivot-3 fact)
    n_tp_per_org = obs.groupby("organoid_id", observed=True)["age_days"].nunique()
    multi_tp = int((n_tp_per_org >= 2).sum())
    total_org = int(obs["organoid_id"].nunique())
    questions.append({
        "id": "A10",
        "question": "How many organoids (bio_samples) in the HNOCA dorsal-telencephalon subset are sampled at two or more distinct timepoints, out of how many total organoids?",
        "expected_tool": None,
        "expected_answer": {"multi_timepoint": multi_tp, "total_organoids": total_org},
        "tolerance": 0,
        "notes": "This is a known structural fact about HNOCA: each bio_sample is typically one snapshot.",
    })

    # ----- classifier-based questions -----
    # A11: classifier output on Velasco D100 (should be > 80% Dorsal Telencephalic Neuron)
    questions.append({
        "id": "A11",
        "question": "If you classify Velasco-protocol cells at day 100 using the HNOCA-trained coarse-label classifier, what is the predicted fraction of 'Dorsal Telencephalic Neuron'?",
        "expected_tool": "classify_cell_type",
        "expected_answer": {"label": "Dorsal Telencephalic Neuron", "min_fraction": 0.75},
        "tolerance": None,
        "notes": "Floor only — accept if predicted fraction is at least 0.75.",
    })

    # ----- write -----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        yaml.safe_dump({"questions": questions}, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {len(questions)} questions to {OUT_PATH}")
    # Echo a few for sanity:
    for q in questions[:3]:
        print(json.dumps(q, indent=2, default=str)[:400])


if __name__ == "__main__":
    main()
