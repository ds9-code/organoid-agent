"""
Download a small real-HNOCA subset for the agent and benchmark layers.

The full cleaned HNOCA atlas is ~17.5 GB.  We stream it from Zenodo over HTTP
range reads and save only a region-filtered, stratified ~5k-cell slice to
``data/hnoca_dt_subset.h5ad`` (~80 MB).  Run this once:

    python download_data.py

To pick a different region or size, call the underlying script directly:

    python scripts/download_hnoca_subset.py --region "Ventral telencephalon" \\
        --n_cells 3000 --out data/hnoca_vt_subset.h5ad
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "hnoca_dt_subset.h5ad"


def main() -> int:
    if DEFAULT_OUT.exists():
        size_mb = DEFAULT_OUT.stat().st_size / 1e6
        print(f"Subset already present at {DEFAULT_OUT} ({size_mb:.1f} MB). "
              "Delete it to re-download.")
        return 0

    script = PROJECT_ROOT / "scripts" / "download_hnoca_subset.py"
    if not script.exists():
        print(f"ERROR: missing {script}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, str(script),
        "--region", "Dorsal telencephalon",
        "--n_cells", "5000",
        "--out", str(DEFAULT_OUT),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
