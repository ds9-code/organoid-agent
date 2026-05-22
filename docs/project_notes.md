# Organoid Agent — Project Notes

## Overview

Build a natural-language agent that uses the **Human Neural Organoid Cell
Atlas** ([HNOCA, He et al. *Nature* 2024](https://www.nature.com/articles/s41586-024-08172-8))
as a tool: ask it a biology question in plain English (*"what cell types do I
get from a Velasco day-100 cortical organoid?"*) and it answers by querying
real HNOCA cells, not by guessing from text.

End goal: same pattern as [mims-harvard/cellflow](https://github.com/mims-harvard/cellflow)
(LLM as the reasoning layer, an ML model as the tool), but for organoid
**benchmarking + discovery**, not for morphogen perturbation prediction.

Two-stage payoff:

1. **Benchmark on HNOCA.** Define 3–5 well-posed tasks; show a foundation
   model (Geneformer / scGPT / scFoundation / UCE) beats classical baselines.
2. **Discover on Paola's organoids.** Project them into the validated HNOCA
   latent space; flag unmatched populations + trajectory deviations.

---

## Thought Process

### 1) Initial Plan (what was already in the repo)

a) **Data.** Pull cleaned HNOCA from Zenodo
   ([record `14161275`](https://zenodo.org/records/14161275)), subset to a
   region / protocol family, build benchmarks on it.

b) **Experiment.** Five candidate tasks defined in `docs/tasks.md`:

```
T1  next-timepoint pseudobulk      x_t  → x_{t+1}      Pearson / MSE
T2  out-of-protocol cell-type      X    → label        macro-F1 / accuracy
T3  primary-reference fidelity     cell → score        OT distance / sim
T4  perturbation response          ctrl → treated      composition KL
T5  real-time pseudotime           cell → age_days     Spearman / Kendall
```

Each baseline implements `BaselineModel.fit / predict`; tasks plug into the
same `Task.prepare → score` interface. Foundation models drop in behind the
same API.

c) **Figures / Metrics.** A leaderboard per task + a pred-vs-true scatter for
the regression tasks.

**Diagram (as cloned):**

```
make_synthetic_hnoca()  →  Task.prepare  →  Model.fit/predict  →  metrics
   (fake counts)            (T1/T2/T5)        (identity, ridge,    (Pearson,
                                                logreg, KNN, PC1)    F1, ρ)
                                              │
                                              ▼
                                       plots/*.png + results_long.csv
```

> **Pivot 1.** The plots advertised as "HNOCA benchmarks" were computed on
> `make_synthetic_hnoca()` — a generator that mimics the obs schema but
> invents counts. README claimed real numbers; nothing in the repo ever
> touched Zenodo. So the leaderboards (T1 Linear Pearson 0.995, etc.) were
> self-congratulatory smoke tests, not benchmarks.

> **Pivot 2.** Real HNOCA is 17.5 GB (cleaned) → 49 GB (full). Too big to just
> "download and iterate". The `disease_atlas.h5ad` on the same record is only
> 2 GB but is a disease sub-atlas — different scope. HNOCA-tools doesn't ship
> a downsampled example. Need a different access pattern.

### 2) Modified Plan — stream just what we need

a) **Data.** The cleaned h5ad is HDF5 with a chunked sparse-CSR `/X`. You can
   open it remotely with `fsspec`+`h5py`, read only `/obs` to pick cells, and
   read only those rows from `/X`. Total bandwidth: **~80 MB instead of 17.5
   GB**.
   - Filter to one region (default `Dorsal telencephalon`, 757k available).
   - Stratify-sample 5k cells across timepoints.
   - Rename obs columns to the repo's schema:
     `annot_region_rev2 → region`, `organoid_age_days → age_days`,
     `assay_differentiation → protocol`, `bio_sample → organoid_id`.

b) **Experiment.** Same T1 / T2 / T5 with the same baselines, but on the real
   `~4,430 cell × 36,842 gene` subset.

c) **Figures / Metrics.** Same panels; the regenerated numbers are honest now
   (table below).

**Diagram (streamed):**

```
Zenodo .h5ad  ──fsspec HTTP range reads──▶  h5py.File(remote, "r")
(17.5 GB                                          │
 cleanedmeta)                                     ├──▶ read /obs columns (~50 MB)
                                                  ├──▶ pick row_idx (region + timepoint)
                                                  └──▶ read /X CSR rows in bounded slabs
                                                          │
                                                          ▼
                                          data/hnoca_dt_subset.h5ad (~80 MB)
                                                          │
                                  ┌───────────────────────┴───────────────────────┐
                                  ▼                                                ▼
                          explore_hnoca.py                                01_explore.py
                          (sectioned printout)                            (plots + leaderboards)
```

> **Pivot 3.** Wired the real subset in. T1 immediately broke: it grouped by
> `organoid_id`, but in HNOCA each `bio_sample` = one snapshot. **Only 1 of
> 206 organoids had ≥2 timepoints.** That's a biology-shaped fact about the
> dataset, not a bug in the loader: HNOCA pools cells across studies and the
> time signal is *between* organoids of different ages, not *within*. Switched
> T1 to group by `protocol` → adjacent pseudobulks within a protocol →
> cohort-level trajectory. Now T1 has 13/16 protocols usable.

**Real-data numbers (honest, after the fix):**

| Task | Model | Metric | **Real HNOCA** | (Synthetic) |
|---|---|---|---:|---:|
| T1 | identity | Pearson | **0.48** | 0.974 |
| T1 | pop_mean | Pearson | 0.46 | 0.807 |
| T1 | linear | Pearson | **0.19** | 0.995 |
| T1 | pop_mean | MSE | **0.26** | — |
| T2 | logreg | accuracy | 0.52 | — |
| T2 | logreg | macro-F1 | **0.16** | 0.385 |
| T2 | knn | accuracy | 0.31 | — |
| T5 | PC1 | Spearman | **0.25** | 0.874 |

Takeaways: ridge **loses to identity** on T1 (over-fits 54 cohort pairs ×
500 HVGs); PopMean has the best MSE (between-protocol variance dominates);
PC1 on real data hits 0.25 vs synthetic's 0.87 because protocol/batch
swamps the time axis in PC1. **These are exactly the gaps a foundation
model should close** — and the synthetic data was hiding them.

> **Pivot 4.** Re-read the repo name: `organoid-agent`. There is no agent.
> No `agent/` directory, no LLM call, no tool-calling, no OpenAI SDK
> anywhere. The repo is a benchmark scaffold pretending to be an agent.
> Looked at [mims-harvard/cellflow](https://github.com/mims-harvard/cellflow)
> as the reference for what an "agent that uses an ML model as a tool" should
> look like.

### 3) New Plan — cellflow-style agent layer

a) **Data.** Same as Modified Plan (streamed subset).

b) **Tool layer.** `agent/hnoca_model.py` — `HNOCAModel` loads the subset
   once, fits log1p → top-2k HVG → 30-D PCA → kNN, and exposes four verbs.
   **Every numeric answer comes from real cells**; the LLM is forbidden from
   inventing.

   | Tool | What it returns |
   |---|---|
   | `query_composition(protocol, age_min, age_max)` | Cell-type composition of cells matching a filter |
   | `gene_expression_timecourse(gene, protocol)` | Mean log1p expression by age bin |
   | `predict_composition_at_age(protocol, target_age_days)` | kNN-retrieved composition at nearby ages |
   | `find_similar_cells(protocol, age_days, k)` | Latent-space neighbours: which protocols + cell types this query sits near |

   Plus `describe_inputs()` so the system prompt is built from the *live*
   valid protocols / cell types / age range — the LLM can't pick an invalid
   protocol because the prompt only lists real ones.

c) **Agent layer.** `agent/agent.py` — OpenAI SDK chat loop with
   function-calling. Interactive REPL **or** one-shot:
   `python -m agent.agent "What dominates Velasco day-100?"`. Works against
   any OpenAI-compatible endpoint (set `OPENAI_BASE_URL`).

d) **Figures / Metrics.** Same `01_explore.py` benchmark plots + annotated
   transcripts in `demo/example_session.md`.

**Diagram (current architecture):**

```
User question  ──▶  OpenAI SDK chat  ──▶  tool call (JSON)  ──▶  HNOCAModel
("what cells in     (function-calling)    {"protocol":           │
 Velasco day-100?")                        "Velasco", ...}        │
                          ▲                                       │
                          │                                       ▼
                          │                              real HNOCA cells
                          │                              (data/hnoca_dt_subset.h5ad)
                          │                                       │
                          │                                       ▼
                          │                              composition / timecourse /
                          │                              neighbours (real numbers)
                          │                                       │
                          └───────────── tool result ◀────────────┘
                          │
                          ▼
                  biological narration
                  ("90% dorsal telencephalic neurons,
                   dominated by cortical pyramidal...")
```

### Deferred (Phase 2)

- T3 (primary-reference fidelity) — needs HNOCA-tools' per-cell similarity
  scores; not in obs of the cleaned h5ad subset I streamed.
- T4 (perturbation response) — needs a clean control-vs-treated split, which
  isn't a first-class field in HNOCA (it's a development atlas, not a
  perturbation screen).
- Plug in a real foundation model (Geneformer / scGPT / UCE) as another
  `BaselineModel`.
- Project Paola's organoids into the HNOCA latent (HNOCA-tools `map_query`)
  → flag (i) unmatched cell populations, (ii) trajectory deviations from the
  closest atlas protocol.

---

## Understanding HNOCA

**Headline.** Integrated transcriptomic atlas of human neural organoids:
**~1.7M cells**, **36 datasets**, **26 differentiation protocols** (3 unguided
+ 23 guided), organoid ages **day 7 → day 450**. Mapped to a curated primary
developing-brain reference for fidelity scoring. Built with scanpy 1.9.3,
integrated with scVI / scANVI, time-aware pseudotime via moscot OT.

**Files on Zenodo `14161275`:**

| File | Size | What it is |
|---|---:|---|
| `hnoca_cleanedmeta.h5ad` | 17.5 GB | Cleaned atlas, what we stream from |
| `hnoca_extended.h5ad` | 18.8 GB | Cleaned + extra embeddings |
| `disease_atlas.h5ad` | 2.0 GB | Disease sub-atlas; different scope |

**Real obs schema (selected — there are ~50 columns):**

```
annot_region_rev2       Dorsal telencephalon / Ventral telencephalon / Medulla / ...
annot_level_2           Dorsal Telencephalic Neuron / NPC / IP / Astrocyte / OPC / ...
annot_level_3_rev2      Finer subdivision under level_2
organoid_age_days       float, days post-induction (7 - 450)
assay_differentiation   Full protocol citation (e.g. "Velasco, 2019 (doi: ...)")
bio_sample              Organoid ID (≈ one organoid at one timepoint)
batch                   Sequencing batch
publication             First-author short
cell_type               cellxgene-harmonised label (e.g. "cerebral cortex pyramidal neuron")
```

**X is sparse CSR**: `/X/data` 3.92 G float32, `/X/indices` 3.92 G int64,
`/X/indptr` 1.77 M int64. Gzip-compressed, chunk size 59,810. This is what
makes the streaming subset feasible — you only fetch the chunks containing
your selected rows.

**Region distribution (full atlas):**

```
Dorsal telencephalon     757,626 cells   ← what we subset to
Unspecific               369,569
Ventral telencephalon    158,506
Medulla                  144,802
Cerebellum                99,577
Thalamus                  73,377
Pons                      54,739
...
```

**Our Dorsal-telencephalon subset (4,430 cells, 16 protocols, days 15–300):**

```
Cell type (cellxgene harmonised)                                 % of subset
cerebral cortex pyramidal neuron                                       17.3%
unknown                                                                14.1%
radial glial cell                                                      13.3%
neuroblast (sensu Vertebrata)                                          12.2%
pyramidal neuron                                                       12.1%
cerebral cortex neuron                                                  6.2%
extratelencephalic-projecting glutamatergic cortical neuron             6.1%
glutamatergic neuron                                                    3.8%
(28 unique cell types total)

annot_level_2 (coarse class):
Dorsal Telencephalic Neuron                                            68.6%
Dorsal Telencephalic NPC                                               29.8%
Dorsal Telencephalic IP                                                 1.6%
```

Why this subset is a good benchmark substrate:

- **Real time axis** (15–300 d) → T1 / T5 are well-posed.
- **16 protocols** (Velasco, Lancaster, Pasca, Yoon, Watanabe, Quadrato,
  Trujillo, Bhaduri, Miura, Andersen, Pellegrini, Esk, Huang, ...) →
  out-of-protocol generalization is testable.
- **Primary reference annotations** in the full atlas → T3 is possible later.

---

## Useful Commands

### Conda / env

```bash
conda create -n organoid-agent python=3.11 -y
conda activate organoid-agent
pip install -r requirements.txt
```

### Streaming an h5ad over HTTPS (fsspec + h5py)

```python
import fsspec, h5py, aiohttp
url = "https://zenodo.org/api/records/14161275/files/hnoca_cleanedmeta.h5ad/content"
timeout = aiohttp.ClientTimeout(total=600, sock_read=300, sock_connect=60)
fs = fsspec.filesystem("https", client_kwargs={"timeout": timeout})
f = fs.open(url, mode="rb", block_size=8 * 1024 * 1024)   # 8 MB cache blocks
hf = h5py.File(f, "r")                                    # remote, no full download
list(hf.keys())                                           # ['X', 'obs', 'var', 'uns', ...]
list(hf["obs"].keys())                                    # all obs columns
```

> Read a *categorical* obs column with `g['codes'][:]` + decoded `g['categories'][:]`.

### Inspecting a CSR layout before reading

```python
for k in ['X/data', 'X/indices', 'X/indptr']:
    d = hf[k]
    print(k, d.shape, d.dtype, "chunks", d.chunks, "compression", d.compression)
```

Tells you how big the chunks are → how much you'll fetch per read.

### GitHub from the CLI

```bash
gh api repos/OWNER/REPO/contributors --jq '.[] | "\(.login)\t\(.contributions)"'
gh api repos/OWNER/REPO/stats/contributors                      # async recompute trigger
gh api repos/OWNER/REPO/commits/<sha> --jq '.commit.author.name + " | " + .commit.message'
```

When a co-author trailer gets stuck in the sidebar UI: amend the commit to
drop the trailer, force-push, hit `/stats/contributors` once or twice, wait
~24 h for the widget cache.

### Useful Python introspection on AnnData

| Command | What it does |
|---|---|
| `adata` | One-line shape + obs/var/uns/obsm/obsp summary |
| `adata.obs.columns.tolist()` | All cell-level metadata fields |
| `adata.obs["region"].value_counts()` | Histogram a categorical column |
| `adata.obs.groupby("protocol", observed=True)["age_days"].nunique()` | Time-coverage per protocol (this is the check that revealed Pivot 3) |
| `adata.X[0, :10].toarray()` | Peek at first cell's first 10 genes |
| `adata.var_names.get_loc("NEUROD6")` | Gene-symbol → column index |
| `adata.uns.keys()` | Run-level metadata (HVG, log1p params, neighbour graph, ...) |

### Project-specific shortcuts (Makefile)

```bash
make data        # stream the ~80 MB subset from Zenodo
make explore     # sectioned printout of what's in the subset
make plots       # regenerate plots/*.png from real cells
make agent       # interactive REPL  (needs OPENAI_API_KEY)
make demo        # one-shot agent question
make test        # pytest sanity checks
```

---

## How to reproduce this from scratch

1. **Clone.**
   ```bash
   git clone https://github.com/ds9-code/organoid-agent.git
   cd organoid-agent
   ```

2. **Env.**
   ```bash
   conda create -n organoid-agent python=3.11 -y
   conda activate organoid-agent
   pip install -r requirements.txt
   ```

3. **Get the data subset** (~80 MB, ~30 s on a decent connection).
   ```bash
   python download_data.py
   # = python scripts/download_hnoca_subset.py --region "Dorsal telencephalon" \
   #       --n_cells 5000 --out data/hnoca_dt_subset.h5ad
   ```
   - This streams the cleaned HNOCA atlas from Zenodo over HTTP range reads.
     **You do not download 17.5 GB.** No HF token, no auth.
   - If it stalls on a slab, retry — script has 3-attempt retry per slab but
     occasional Zenodo hiccups happen.

4. **Sanity-check what's in it.**
   ```bash
   python explore_hnoca.py     # 8 sectioned printouts: shape, X sparsity,
                               # obs columns, protocols, ages, cell types,
                               # var, uns
   ```

5. **Regenerate the plots.**
   ```bash
   make plots                  # writes plots/*.png + results_long.csv +
                               # results.json — real HNOCA numbers
   ```

6. **(Optional) Run the agent.** Needs an OpenAI-style API key.
   ```bash
   cp .env.example .env        # fill in OPENAI_API_KEY (and optionally
                               # OPENAI_BASE_URL / AGENT_MODEL)
   make demo                   # one example question
   make agent                  # interactive REPL
   ```

### For pushing changes (Github)

```bash
git status                                    # eyeball untracked vs gitignored
git add <files>                               # prefer specific names over -A
git commit -m "..."                           # one-liner if scope is small
                                              # HEREDOC for multi-paragraph
git push                                      # to existing remote
```

`.env`, `data/*.h5ad`, `plots/*.png`, `plots/*.csv`, `plots/*.json`,
`__pycache__/` are gitignored. **Check `git status --porcelain` before push** —
once a secret is in a commit it's a pain to remove.

If you want a clean co-author trailer (or to remove one):
- Add: end the commit body with
  `Co-Authored-By: Name <email@example.com>`
- Remove from HEAD: `git commit --amend -m "..."` without the trailer, then
  `git push --force-with-lease`. The sidebar contributor widget on GitHub
  takes ~24 h to refresh even after the API is clean.

---

## Project Notes p2

### Open questions

1. **Which foundation model first?** Candidates: Geneformer (BERT-style,
   gene-token), scGPT (auto-regressive, more flexible), scFoundation (large
   pretrained), UCE (universal cell embedding). Geneformer is the simplest
   pip install + `tokenize → embed → linear head`; probably the right first
   pin so we get *some* leaderboard signal fast, then iterate.

2. **Are the T1 numbers actually meaningful?** With group_key=`protocol`
   we have only 13 protocols, → ~54 train pairs, 5 eval pairs. That's a
   tiny eval set. **The Pearson 0.48 for identity is plausible but
   high-variance.** Need to either bootstrap CIs or restrict to protocols
   with denser timepoint coverage (Velasco has 8+ ages, Lancaster 6+, most
   others have 2–3).

3. **What should "novel cell state" mean for Paola's organoids?** Two
   defensible operationalisations:
   - **Latent-space outlier**: mean distance to k nearest HNOCA cells >
     some percentile of intra-HNOCA distances.
   - **Trajectory deviation**: predicted age (from a regressor trained on
     HNOCA) ≠ actual culture age.

   Need to pick one before we have data.

4. **Cell-type label space is messy.** 28 fine labels in the subset; many
   are nearly-synonymous (`cerebral cortex pyramidal neuron` vs
   `pyramidal neuron`). T2's macro-F1 of 0.16 partly reflects this. The
   `annot_level_2` coarse labels (3 classes for dorsal telencephalon) give
   a cleaner T2 — should probably make `label_col` a config knob and
   evaluate both.

5. **Does CellFlow's flow-matching approach transfer to HNOCA?** CellFlow
   was trained on **perturbation pairs** (control → treated). HNOCA has
   **developmental trajectories** (young → old, same protocol). Different
   structure. Worth thinking about whether the OT-based formulation works
   here or if a simpler regression is more honest.

### Things worth doing soon

- Add `--label_col` to T2 so we can compare fine vs coarse cell-type F1.
- Compute T1 with bootstrap CIs (n=500 resamples of held-out protocols).
- Stream a second subset (Ventral telencephalon, ~158k available cells) →
  test cross-region generalization.
- Hook up Geneformer as a `BaselineModel`. The wrapper is mostly
  `tokenize → embed → ridge head`.
- T3: add the primary-reference similarity score column when streaming the
  subset. (It's in the full atlas's obs but I didn't pull it the first time.)
- T4: the only "perturbation-like" axis in HNOCA is *protocol* itself —
  could frame T4 as "predict the composition shift between Velasco and
  Lancaster at matched ages." Not a true perturbation but a real signal.

### Things to bring up with the lab

- Lock the first benchmark region. Dorsal telencephalon is the biggest and
  best-annotated; sticking with it for v1 unless someone has a reason to
  expand.
- Time-course vs snapshot on Paola's side — **T1 / T5 need time-course**
  cultures, **T2 / T3 work on snapshots**. What does Paola actually have?
- Pick the first foundation model (Geneformer is my vote).
- For T4: is there a planned perturbation arm in Paola's dataset, or are we
  always operating on the protocol-as-perturbation framing?

---

## misc

- **Surprising finding from `explore_hnoca.py`**: 14% of cells in the Dorsal
  telencephalon subset are labelled `"unknown"` in `cell_type` — second
  most common class. That's not noise, that's a real "cellxgene didn't
  harmonise this protocol's labels" gap. Worth flagging when reporting T2.
- **Surprising finding from T5**: PC1 Spearman of 0.25 means **PC1 is not
  primarily a time axis** in this subset. Either protocol or batch
  dominates. A protocol-conditional PC1 (Spearman within each protocol,
  then averaged) would be a much fairer baseline — added to the worth-doing
  list.
- **The HNOCA file structure trick**: `/obs` is small enough that you can
  read every categorical column into memory in <30 s of HTTPS traffic.
  That's the unlock — once you have all obs locally, you can pick exactly
  which rows to fetch and never touch the full 17.5 GB.
- **Why pop-mean wins MSE on T1**: when between-protocol variance >>
  within-protocol-trajectory variance, the protocol mean is closer to any
  held-out protocol than identity is. Tells you the model needs to condition
  on protocol identity, not just on `x_t`.
- **`Co-Authored-By` trailer thing**: when you commit with a trailer naming
  an email that's claimed by a GitHub account, that account shows up in the
  contributors graph. Amending the commit + force-pushing removes them from
  the live API immediately, but the right-sidebar widget cache takes hours
  to a day. The `/graphs/contributors` page hits the live API and updates
  faster.
- **HuggingFace is overkill for HNOCA**: cellflow uses HF for the iNeurons
  dataset because it's gated. HNOCA on Zenodo is **public**, so we never
  need an `HF_TOKEN`. The `.env.example` keeps the OpenAI-related vars only.

---

## Repo layout (current)

```
organoid-agent/
├── README.md                 # cellflow-style: example session, How it works, Setup
├── Makefile                  # data / explore / plots / agent / demo / test
├── pyproject.toml
├── requirements.txt
├── .env.example              # OPENAI_API_KEY / OPENAI_BASE_URL / AGENT_MODEL
├── download_data.py          # one-call subset fetcher
├── explore_hnoca.py          # sectioned printout of the real subset
├── agent/
│   ├── __init__.py
│   ├── hnoca_model.py        # the tool — 4 verbs over real HNOCA cells
│   └── agent.py              # OpenAI SDK chat loop with function-calling
├── scripts/
│   └── download_hnoca_subset.py   # streaming HTTP range reads (fsspec + h5py)
├── src/
│   ├── data/{hnoca,synthetic}.py
│   ├── tasks/{base,t1_next_timepoint,t2_celltype_oop,t5_pseudotime}.py
│   ├── models/{base,baselines}.py
│   ├── eval/{metrics,runner}.py
│   └── plot/figures.py
├── notebooks/01_explore.py   # generates plots/ from the real subset
├── plots/                    # PNGs + CSV/JSON (gitignored)
├── data/                     # downloaded subsets (gitignored)
├── configs/                  # YAML configs per task slice
├── docs/
│   ├── hnoca_summary.md
│   ├── tasks.md
│   ├── decisions.md
│   ├── week1_summary.md
│   └── project_notes.md      ← this file
├── demo/example_session.md   # annotated agent transcript
└── tests/test_smoke.py       # pytest sanity checks (still on synthetic, by design)
```
