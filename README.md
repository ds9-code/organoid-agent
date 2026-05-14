# HNOCA Organoid Agent

A natural-language agent (built on the OpenAI Python SDK) that uses a small
slice of the [**Human Neural Organoid Cell Atlas**](https://www.nature.com/articles/s41586-024-08172-8)
(He, Dony, Fleck et al., *Nature* 2024) as a tool: ask "what cells dominate a
Velasco day-100 cortical organoid?" and the agent queries real HNOCA cells,
then explains the predicted composition biologically.

The benchmark layer underneath (`src/`) implements 5 candidate tasks (next-
timepoint, out-of-protocol cell-type, primary-reference fidelity, perturbation,
real-time pseudotime) so any foundation model — Geneformer, scGPT,
scFoundation, UCE — can be plugged in behind the same interface and compared
against classical baselines on real HNOCA data.

```
You: What cells dominate a Velasco day-100 cortical organoid?
  -> calling query_composition({"protocol": "Velasco", "age_min": 80, "age_max": 120})
Agent: ~91% dorsal-telencephalic neurons (real HNOCA cells, n=889 across 42
       organoids in this window) — dominated by cerebral-cortex pyramidal
       (43.3%), extratelencephalic-projecting glutamatergic cortical (23.1%),
       and pyramidal (18.9%) sub-identities, with a residual ~4% radial-glia
       pool. Consistent with mid-stage Velasco-protocol cortical organoids
       being mostly post-mitotic deep-layer cortical neurons.
```

See [`demo/example_session.md`](demo/example_session.md) for a full annotated
transcript, or run `make demo` once set up.

## How it works

| Layer | File | Role |
|---|---|---|
| Data | [`scripts/download_hnoca_subset.py`](scripts/download_hnoca_subset.py) | Streams the cleaned HNOCA atlas (17.5 GB on Zenodo) over HTTP range reads with `fsspec`+`h5py`, reads only `/obs` and a stratified slice of CSR rows, and saves a ~80 MB region-filtered `.h5ad` locally. |
| Tool (ML / retrieval) | [`agent/hnoca_model.py`](agent/hnoca_model.py) | `HNOCAModel` loads the subset once, computes log1p + 30-D PCA + a kNN index, and exposes four tools: `query_composition`, `gene_expression_timecourse`, `predict_composition_at_age`, `find_similar_cells`. All answers come from real cells; the LLM never invents numbers. |
| Reasoning | [`agent/agent.py`](agent/agent.py) | An LLM chat loop (OpenAI SDK) that exposes the above as function-calling tools, maps loose requests ("a Velasco organoid around 3 months") to concrete `(protocol, age)` pairs, and explains the results. Point `OPENAI_BASE_URL` at any OpenAI-compatible endpoint. |
| Benchmark | [`src/`](src/) and [`notebooks/01_explore.py`](notebooks/01_explore.py) | 3 task implementations (T1 next-timepoint, T2 out-of-protocol cell-type, T5 pseudotime) × 6 baselines (identity, pop-mean, ridge, logreg, kNN, PC1) on real HNOCA data, with metrics + leaderboard PNGs in `plots/`. |
| Eval harness | [`src/eval/`](src/eval/) | Tiny task/metric/runner harness so foundation-model wrappers plug into the same `BaselineModel` interface. |

## Setup

```bash
# 1. Environment
conda create -n organoid-agent python=3.11 -y
conda activate organoid-agent
pip install -r requirements.txt

# 2. Secrets (only needed to run the agent layer; the benchmark works without it)
cp .env.example .env
#   OPENAI_API_KEY=sk-...
#   OPENAI_BASE_URL=...   (optional; for Dartmouth: https://chat.dartmouth.edu/api)
#   AGENT_MODEL=...       (default gpt-4o)
```

There's a `Makefile` with shortcuts: `make data`, `make explore`, `make plots`,
`make agent`, `make demo`, `make test`.

### Data access

`python download_data.py` streams a ~80 MB Dorsal-telencephalon subset of the
cleaned HNOCA atlas from Zenodo (record `14161275`) to
`data/hnoca_dt_subset.h5ad`. The trick: HNOCA is HDF5 with chunked sparse-CSR
`/X`, so we open it remotely with `fsspec`+`h5py`, read only `/obs` to pick
~5,000 cells stratified by timepoint, and read only those CSR rows in bounded
slabs. **No full 17.5 GB download is needed**, and no credentials.

To pick a different region or size:

```bash
python scripts/download_hnoca_subset.py \
    --region "Ventral telencephalon" --n_cells 3000 \
    --out data/hnoca_vt_subset.h5ad
```

## Run the agent

```bash
make agent                                                   # interactive REPL
make demo                                                    # one example question
python -m agent.agent "Compare Velasco vs Lancaster at day 60."   # one-shot
```

The first call to `HNOCAModel(...)` takes ~10 s (loads the subset, fits a 30-D
PCA, builds a kNN index); after that each tool call is sub-second.

### Environment variables

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `OPENAI_API_KEY` | yes (agent only) | — | OpenAI-style API key. |
| `OPENAI_BASE_URL` | no | official OpenAI API | OpenAI-compatible endpoint (Dartmouth, vLLM, etc.). |
| `AGENT_MODEL` | no | `gpt-4o` | Model name the endpoint exposes. Proxies often namespace these. |
| `HNOCA_DATA_PATH` | no | `data/hnoca_dt_subset.h5ad` | Override the subset path. |

## Run the benchmark layer (no LLM)

```bash
make plots          # python notebooks/01_explore.py -- regenerates plots/*.png + results CSV/JSON
make test           # pytest sanity checks on synthetic data
```

`notebooks/01_explore.py` loads the real subset, runs T1 / T2 / T5 with all
baselines, and writes 9 PNGs + `results_long.csv` + `results.json` to `plots/`.
Task definitions live in `src/tasks/`, baselines in `src/models/`, the harness
in `src/eval/`. To plug in a foundation model: implement `fit`/`predict` on
`src/models/base.BaselineModel`, add it to the list in `01_explore.py`.

## Using the tool layer directly (no LLM)

```python
from agent.hnoca_model import HNOCAModel

m = HNOCAModel()
print(m.describe_inputs())                                     # valid protocols / ages / cell types
print(m.query_composition(protocol="Velasco", age_min=80, age_max=120))
print(m.gene_expression_timecourse("NEUROD6"))
print(m.predict_composition_at_age("Velasco", target_age_days=100))
print(m.find_similar_cells(protocol="Velasco", age_days=100))
```

## What lives where

- **GitHub** (this repo): all the code — `scripts/`, `agent/`, `src/`,
  `notebooks/`, the `Makefile`, the demo, docs.
- **Zenodo** ([record `14161275`](https://zenodo.org/records/14161275)): the
  full HNOCA atlas. We never commit it; we stream subsets on demand.

## Layout

```
agent/                   # LLM chat loop + HNOCAModel tool
  agent.py
  hnoca_model.py
scripts/
  download_hnoca_subset.py   # streaming subset downloader (HTTP range reads)
src/
  data/                  # AnnData loaders (HNOCA, Paola lab, synthetic)
  tasks/                 # T1 next-timepoint, T2 cell-type OOP, T5 pseudotime
  models/                # baselines (identity, pop-mean, ridge, logreg, kNN, PC1)
  eval/                  # metrics + runner
  plot/                  # plot helpers
configs/                 # YAML configs per task slice
docs/                    # HNOCA summary, task spec, decisions log, week-1 summary
notebooks/01_explore.py  # generates plots/ from the real subset
plots/                   # output figures + results CSV/JSON (gitignored)
data/                    # downloaded subsets (gitignored)
demo/example_session.md  # annotated agent transcript
tests/test_smoke.py      # pytest sanity checks
download_data.py         # one-call subset downloader
explore_hnoca.py         # printed summary of what's in the subset
Makefile
.env.example
```

## Roadmap

- [x] Real HNOCA wired in; plots come from real cells, not synthetic.
- [x] Agent layer with 4 tools answering from real cells.
- [ ] Plug in a foundation model (Geneformer / scGPT / UCE) as another `BaselineModel`.
- [ ] T3 (primary-reference fidelity) and T4 (perturbation response) tasks.
- [ ] Project Paola's organoids into the HNOCA latent space and flag novel populations.
