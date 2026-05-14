"""
Exploration script for the HNOCA subset.

Run with:
    python explore_hnoca.py

Data location:
    data/hnoca_dt_subset.h5ad   (Dorsal-telencephalon stratified subset,
                                 ~80 MB; streamed from Zenodo with
                                 scripts/download_hnoca_subset.py)

If you don't have it yet:
    python download_data.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import anndata as ad

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "hnoca_dt_subset.h5ad"


def sep(title: str, n: int = 60) -> None:
    print("=" * n)
    print(title)
    print("=" * n)


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found.\n"
              "Run: python download_data.py")
        return 1

    print(f"Loading HNOCA subset from {DATA_PATH} ...")
    adata = ad.read_h5ad(DATA_PATH)
    print("Done.\n")

    sep("1. BASIC SHAPE")
    print(f"  Cells x Genes: {adata.shape}")
    print(f"  ({adata.n_obs:,} cells, {adata.n_vars:,} genes)")
    print(f"  Source: {adata.uns.get('source', 'unknown')}")
    print(f"  Subset region: {adata.uns.get('subset_region', 'unknown')}\n")

    sep("2. EXPRESSION MATRIX (adata.X)")
    X = adata.X
    print(f"  Type    : {type(X).__name__}  (sparse = most entries are 0)")
    print(f"  Dtype   : {X.dtype}  (raw counts)")
    if hasattr(X, "nnz"):
        sparsity = 1 - X.nnz / (adata.shape[0] * adata.shape[1])
        print(f"  Sparsity: {sparsity:.1%} zeros, nnz = {X.nnz:,}")
    row0 = X[0, :12]
    row0 = row0.toarray().ravel() if hasattr(row0, "toarray") else np.asarray(row0).ravel()
    print(f"  First cell, first 12 genes: {row0}\n")

    sep("3. CELL METADATA (adata.obs)")
    print(f"  {len(adata.obs.columns)} columns. All names:")
    cols = adata.obs.columns.tolist()
    for i in range(0, len(cols), 4):
        print("   ", cols[i:i + 4])
    print()

    sep("4. PERTURBATION / EXPERIMENTAL AXES")
    # Region (we already pre-filtered, but show it)
    print("  Region (annot_region_rev2 -> obs['region']):")
    for r, n in adata.obs["region"].value_counts().items():
        if n == 0:
            continue
        print(f"    {r:<28s}  {n:>6,} cells")
    print()

    print("  Differentiation protocol (assay_differentiation -> obs['protocol']):")
    pc = adata.obs["protocol"].value_counts().head(12)
    for p, n in pc.items():
        print(f"    {p[:50]:<50s}  {n:>6,} cells")
    print(f"    ... ({adata.obs['protocol'].nunique()} unique protocols total)\n")

    print("  Organoid age (organoid_age_days -> obs['age_days']):")
    ages = adata.obs["age_days"]
    print(f"    range : {ages.min():.0f} - {ages.max():.0f} days")
    print(f"    median: {ages.median():.0f} days, n unique timepoints: {ages.nunique()}")
    print(f"    top timepoints by cell count:")
    for t, n in ages.value_counts().head(8).items():
        print(f"      day {int(t):>4d}: {n:>5,} cells")
    print()

    print("  Organoid id (bio_sample -> obs['organoid_id']):")
    print(f"    {adata.obs['organoid_id'].nunique()} unique organoids in subset")
    org_n = adata.obs.groupby("organoid_id", observed=True).size()
    print(f"    cells per organoid: min={org_n.min()}, median={int(org_n.median())}, "
          f"max={org_n.max()}\n")

    sep("5. CELL-FATE OUTCOMES (HNOCA annotation)")
    print("  cell_type (cellxgene-harmonised; HNOCA's atlas label):")
    for ct, n in adata.obs["cell_type"].value_counts().head(15).items():
        pct = 100 * n / adata.n_obs
        print(f"    {ct[:55]:<55s}  {n:>6,} cells  ({pct:4.1f}%)")
    print(f"    ... ({adata.obs['cell_type'].nunique()} unique cell types)\n")

    if "annot_level_2" in adata.obs.columns:
        print("  HNOCA annot_level_2 (coarse cell class):")
        for ct, n in adata.obs["annot_level_2"].value_counts().head(12).items():
            print(f"    {ct[:40]:<40s}  {n:>6,} cells")
        print()

    sep("6. GENE METADATA (adata.var)")
    print(f"  Columns: {adata.var.columns.tolist()}")
    print(f"  First 10 genes: {adata.var_names[:10].tolist()}")
    if "gene_symbol" in adata.var.columns:
        print(f"  First 10 symbols: {adata.var['gene_symbol'].iloc[:10].tolist()}")
    print()

    sep("7. UNSTRUCTURED (adata.uns)")
    for k, v in adata.uns.items():
        print(f"  {k}: {type(v).__name__}")
    print()

    sep("8. SAMPLE ROWS (key columns)")
    key_cols = [c for c in ("organoid_id", "age_days", "region", "protocol",
                            "cell_type", "annot_level_2", "publication")
                if c in adata.obs.columns]
    print(adata.obs[key_cols].head(10).to_string())
    print()

    print("Exploration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
