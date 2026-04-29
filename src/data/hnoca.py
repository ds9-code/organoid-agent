"""HNOCA loader — placeholder for real download/subset logic.

The real ~1.7M cell object lives on Zenodo (records 14160929 and 14161275).
This loader is intentionally thin: it accepts a local path the user has
already downloaded to, and exposes filters by region, protocol, and time.

Run on a machine with network access:

    # Cleaned dataset (~few GB):
    wget https://zenodo.org/records/14161275/files/hnoca_clean.h5ad

    # Or use HNOCA-tools / cellxgene API.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad


@dataclass
class HNOCALoader:
    """Load an HNOCA AnnData and subset by metadata.

    Parameters
    ----------
    path : str or Path
        Local path to the .h5ad file (cleaned or full).
    """
    path: str | Path

    def load(
        self,
        regions: Optional[Iterable[str]] = None,
        protocols: Optional[Iterable[str]] = None,
        min_day: Optional[float] = None,
        max_day: Optional[float] = None,
        backed: bool = True,
    ) -> ad.AnnData:
        """Return a (possibly subset) AnnData.

        For the full atlas, prefer `backed='r'` and slice before calling
        `.to_memory()` to avoid loading 1.7M cells into RAM.
        """
        adata = ad.read_h5ad(self.path, backed="r" if backed else None)

        mask = None
        if regions is not None:
            m = adata.obs["region"].isin(list(regions))
            mask = m if mask is None else (mask & m)
        if protocols is not None:
            m = adata.obs["protocol"].isin(list(protocols))
            mask = m if mask is None else (mask & m)
        if min_day is not None:
            m = adata.obs["age_days"] >= float(min_day)
            mask = m if mask is None else (mask & m)
        if max_day is not None:
            m = adata.obs["age_days"] <= float(max_day)
            mask = m if mask is None else (mask & m)

        if mask is not None:
            adata = adata[mask]
        return adata.to_memory() if backed else adata
