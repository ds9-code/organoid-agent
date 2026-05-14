"""Stream a small real-HNOCA subset over HTTP range reads and save locally.

Avoids downloading the full ~17.5 GB cleaned atlas. Opens the Zenodo file with
fsspec + h5py, reads /obs to pick ~5k cells (single region, stratified by
timepoint), and reads only those CSR rows.

Usage:
    python scripts/download_hnoca_subset.py
        --region "Dorsal telencephalon"
        --n_cells 5000
        --out data/hnoca_dt_subset.h5ad
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import fsspec
import h5py
import aiohttp
import anndata as ad


URL_CLEANED = (
    "https://zenodo.org/api/records/14161275/files/hnoca_cleanedmeta.h5ad/content"
)


def _decode_str_arr(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind in ("O", "S"):
        return np.array([a.decode() if isinstance(a, bytes) else a for a in arr])
    return arr


def read_obs_col(hf: h5py.File, name: str):
    g = hf["obs"][name]
    if isinstance(g, h5py.Group) and "codes" in g and "categories" in g:
        codes = g["codes"][:]
        cats = _decode_str_arr(g["categories"][:])
        return pd.Categorical.from_codes(codes, cats)
    return _decode_str_arr(g[:])


def stratified_sample(df: pd.DataFrame, time_col: str, n_target: int, rng) -> np.ndarray:
    """Roughly uniform sample across timepoints, capped at availability."""
    tps = df[time_col].dropna().unique()
    tps = np.sort(tps)
    per = max(1, n_target // len(tps))
    picked = []
    for t in tps:
        idx = df.index[df[time_col] == t].to_numpy()
        if len(idx) == 0:
            continue
        k = min(per, len(idx))
        picked.append(rng.choice(idx, size=k, replace=False))
    out = np.concatenate(picked)
    if len(out) > n_target:
        out = rng.choice(out, size=n_target, replace=False)
    return np.sort(out)


def read_csr_rows(hf: h5py.File, row_idx: np.ndarray, max_chunk_nnz: int = 200_000) -> sp.csr_matrix:
    """Read CSR rows from an h5ad-style /X group. row_idx must be sorted ascending.

    Coalesces adjacent ranges but caps each HTTP read at ``max_chunk_nnz`` elements
    so a single fetch never exceeds a few MB.
    """
    Xg = hf["X"]
    indptr_full = Xg["indptr"]
    print("  reading full indptr ...")
    t = time.time()
    indptr_all = indptr_full[:]
    print(f"    indptr read in {time.time()-t:.1f}s")

    starts = indptr_all[row_idx]
    ends = indptr_all[row_idx + 1]
    nnz_per = ends - starts
    total_nnz = int(nnz_per.sum())
    print(f"  {len(row_idx)} rows, total nnz = {total_nnz:,}")

    data_ds = Xg["data"]
    idx_ds = Xg["indices"]

    out_data = np.empty(total_nnz, dtype=data_ds.dtype)
    out_idx = np.empty(total_nnz, dtype=np.int64)
    out_indptr = np.zeros(len(row_idx) + 1, dtype=np.int64)
    np.cumsum(nnz_per, out=out_indptr[1:])

    # Sort rows by their on-disk start position; group into bounded contiguous slabs.
    order = np.argsort(starts, kind="stable")
    s_sorted = starts[order]
    e_sorted = ends[order]

    slabs = []  # list of (s, e, list_of_orig_row_positions)
    cur_s, cur_e = int(s_sorted[0]), int(e_sorted[0])
    cur_members = [int(order[0])]
    for k in range(1, len(s_sorted)):
        rs, re = int(s_sorted[k]), int(e_sorted[k])
        if rs <= cur_e and (re - cur_s) <= max_chunk_nnz:
            cur_e = max(cur_e, re)
            cur_members.append(int(order[k]))
        else:
            slabs.append((cur_s, cur_e, cur_members))
            cur_s, cur_e = rs, re
            cur_members = [int(order[k])]
    slabs.append((cur_s, cur_e, cur_members))
    print(f"  reading in {len(slabs):,} bounded slabs (cap={max_chunk_nnz} nnz each)")

    t = time.time()
    bytes_raw = 0
    last_print = t
    for mi, (s, e, members) in enumerate(slabs):
        # Retry once on transient timeout/network hiccup
        for attempt in range(3):
            try:
                block_data = data_ds[s:e]
                block_idx = idx_ds[s:e]
                break
            except Exception as ex:
                if attempt == 2:
                    raise
                print(f"    slab {mi} attempt {attempt+1} failed ({type(ex).__name__}); retrying...")
                time.sleep(2 + 2 * attempt)
        bytes_raw += (e - s) * (data_ds.dtype.itemsize + idx_ds.dtype.itemsize)
        for orig_pos in members:
            os_, oe_ = int(starts[orig_pos]), int(ends[orig_pos])
            local_s = os_ - s
            local_e = oe_ - s
            ds, de = int(out_indptr[orig_pos]), int(out_indptr[orig_pos + 1])
            out_data[ds:de] = block_data[local_s:local_e]
            out_idx[ds:de] = block_idx[local_s:local_e]
        now = time.time()
        if now - last_print > 5 or mi == len(slabs) - 1:
            print(f"    slab {mi+1}/{len(slabs)} | t={now-t:.1f}s | "
                  f"~{bytes_raw/1e6:.0f}MB uncompressed")
            last_print = now
    print(f"  data/indices done in {time.time()-t:.1f}s")
    return sp.csr_matrix((out_data, out_idx, out_indptr)), total_nnz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Dorsal telencephalon")
    ap.add_argument("--n_cells", type=int, default=5000)
    ap.add_argument("--out", default="data/hnoca_dt_subset.h5ad")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block_mb", type=int, default=4)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print(f"Opening remote h5ad with HTTP range reads (block_size={args.block_mb}MB)...")
    timeout = aiohttp.ClientTimeout(total=600, sock_read=300, sock_connect=60)
    fs = fsspec.filesystem("https", client_kwargs={"timeout": timeout})
    f = fs.open(URL_CLEANED, mode="rb", block_size=args.block_mb * 1024 * 1024)
    hf = h5py.File(f, "r")

    # ---- read obs columns we want
    obs_cols = [
        "annot_region_rev2", "organoid_age_days", "cell_type", "bio_sample",
        "assay_differentiation", "batch", "publication", "annot_level_2",
        "annot_level_3_rev2", "disease",
    ]
    print("Reading obs columns:", obs_cols)
    t0 = time.time()
    obs_data = {}
    for c in obs_cols:
        if c not in hf["obs"]:
            continue
        obs_data[c] = read_obs_col(hf, c)
        print(f"  {c} ✓ ({time.time()-t0:.1f}s)")
    obs = pd.DataFrame({k: pd.Series(v) for k, v in obs_data.items()})

    # ---- subset
    mask = obs["annot_region_rev2"] == args.region
    print(f"\nRegion '{args.region}': {int(mask.sum()):,} cells available")
    if mask.sum() == 0:
        raise SystemExit("No cells match region filter.")
    sub = obs.loc[mask].copy()
    sub.index = np.arange(len(obs))[mask.to_numpy()]  # store original row index

    row_idx = stratified_sample(sub, "organoid_age_days", args.n_cells, rng)
    print(f"Sampled {len(row_idx):,} cells (stratified by organoid_age_days)")

    # ---- read X rows
    print("\nReading var (gene metadata)...")

    def _read_var_col(name):
        g = hf["var"][name]
        if isinstance(g, h5py.Group) and "codes" in g and "categories" in g:
            codes = g["codes"][:]
            cats = _decode_str_arr(g["categories"][:])
            return np.array(pd.Categorical.from_codes(codes, cats))
        return _decode_str_arr(g[:])

    var_index = _decode_str_arr(hf["var"]["_index"][:])
    gene_symbols = _read_var_col("gene_symbol") if "gene_symbol" in hf["var"] else var_index
    n_genes = len(var_index)
    print(f"  n_genes = {n_genes:,}")

    print("\nReading CSR rows for sampled cells...")
    X_sub, total_nnz = read_csr_rows(hf, row_idx)
    # Patch n_cols (we placeholdered above)
    X_sub = sp.csr_matrix((X_sub.data, X_sub.indices, X_sub.indptr), shape=(len(row_idx), n_genes))

    # ---- build adata
    obs_sub = obs.iloc[row_idx].reset_index(drop=True).copy()
    var_df = pd.DataFrame({"gene_symbol": gene_symbols}, index=pd.Index(var_index, name="gene"))

    # rename a few columns to the schema the rest of the repo expects
    obs_sub = obs_sub.rename(columns={
        "annot_region_rev2": "region",
        "organoid_age_days": "age_days",
        "assay_differentiation": "protocol",
        "bio_sample": "organoid_id",
    })

    adata = ad.AnnData(X=X_sub, obs=obs_sub, var=var_df)
    adata.uns["source"] = "HNOCA cleanedmeta (Zenodo 14161275) — streamed subset"
    adata.uns["subset_region"] = args.region
    adata.uns["n_subset_cells"] = int(adata.n_obs)

    print(f"\nWriting {out_path} ...")
    adata.write_h5ad(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Done. shape={adata.shape}, file size={size_mb:.1f} MB")


if __name__ == "__main__":
    main()
