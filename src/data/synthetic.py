"""Synthetic HNOCA-shaped data so the rest of the pipeline can be tested
without network access. Mimics:
  - cells x genes count matrix
  - obs columns: protocol, region, age_days, cell_type, organoid_id
  - a smooth time effect on a subset of "lineage genes" so t->t+1 is non-trivial
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad


_REGIONS = ["telencephalon", "cerebral_cortex", "midbrain", "diencephalon", "hindbrain"]
_PROTOCOLS = [f"protocol_{i:02d}" for i in range(8)]
_CELL_TYPES = [
    "neural_progenitor", "radial_glia", "intermediate_progenitor",
    "excitatory_neuron", "inhibitory_neuron", "astrocyte", "oligodendrocyte",
]


def make_synthetic_hnoca(
    n_cells: int = 6000,
    n_genes: int = 800,
    n_organoids: int = 40,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)

    organoid_ids = np.array([f"org_{i:03d}" for i in range(n_organoids)])
    org_protocol = rng.choice(_PROTOCOLS, size=n_organoids)
    org_region = rng.choice(_REGIONS, size=n_organoids)

    # Each organoid is sampled at multiple timepoints
    timepoints = np.array([15, 30, 60, 90, 150, 250, 400], dtype=float)

    # Distribute n_cells across (organoid, timepoint) combos
    rows = []
    for oi in range(n_organoids):
        for t in timepoints:
            n = max(5, int(rng.poisson(n_cells / (n_organoids * len(timepoints)))))
            rows.append((oi, t, n))
    total = sum(r[2] for r in rows)
    if total > n_cells:
        # Trim
        scale = n_cells / total
        rows = [(o, t, max(2, int(n * scale))) for (o, t, n) in rows]

    obs_records = []
    cell_idx = 0
    for (oi, t, n) in rows:
        for _ in range(n):
            obs_records.append({
                "organoid_id": organoid_ids[oi],
                "protocol": org_protocol[oi],
                "region": org_region[oi],
                "age_days": float(t),
            })
            cell_idx += 1
    obs = pd.DataFrame(obs_records)
    n_cells_actual = len(obs)

    # ------------- Build expression -------------
    # Lineage signal: a small set of "lineage genes" whose mean increases (or decreases) with t.
    n_lineage = 50
    lineage_dir = rng.choice([-1, 1], size=n_lineage)

    base_mean = rng.gamma(1.0, 1.0, size=n_genes) + 0.2
    log_t = np.log1p(obs["age_days"].to_numpy())
    log_t_norm = (log_t - log_t.mean()) / (log_t.std() + 1e-9)

    region_codes = pd.Categorical(obs["region"]).codes
    region_offset = rng.normal(0, 0.4, size=(len(_REGIONS), n_genes))

    # log-normal-like counts
    mu = np.broadcast_to(base_mean, (n_cells_actual, n_genes)).copy()
    mu[:, :n_lineage] *= np.exp(0.7 * log_t_norm[:, None] * lineage_dir[None, :])
    mu += region_offset[region_codes]
    mu = np.clip(mu, 0.05, None)
    counts = rng.poisson(mu * 5.0).astype(np.float32)

    # Cell-type label: derive from a soft mixture of (region, age, lineage signal)
    progenitor_score = -log_t_norm + rng.normal(0, 0.3, n_cells_actual)
    neuron_score = log_t_norm + rng.normal(0, 0.3, n_cells_actual)
    glia_score = np.maximum(0, log_t_norm - 0.6) + rng.normal(0, 0.3, n_cells_actual)

    ctype = np.empty(n_cells_actual, dtype=object)
    for i in range(n_cells_actual):
        scores = {
            "neural_progenitor": progenitor_score[i] + (1.0 if obs["age_days"].iat[i] < 60 else 0),
            "radial_glia": progenitor_score[i] * 0.7,
            "intermediate_progenitor": (progenitor_score[i] + neuron_score[i]) * 0.4,
            "excitatory_neuron": neuron_score[i] + (0.4 if obs["region"].iat[i] in ("cerebral_cortex", "telencephalon") else 0),
            "inhibitory_neuron": neuron_score[i] * 0.8,
            "astrocyte": glia_score[i],
            "oligodendrocyte": glia_score[i] * 0.5 - 0.4,
        }
        ctype[i] = max(scores, key=scores.get)
    obs["cell_type"] = ctype

    var = pd.DataFrame({"gene_symbol": [f"g{i:04d}" for i in range(n_genes)]})
    var.index = var["gene_symbol"]
    var["is_lineage"] = False
    var.iloc[:n_lineage, var.columns.get_loc("is_lineage")] = True

    obs.index = pd.Index([f"cell_{i:06d}" for i in range(n_cells_actual)])
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.uns["timepoints"] = timepoints.tolist()
    adata.uns["source"] = "synthetic_hnoca_shape"
    return adata
