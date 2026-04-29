# Week-1 status

## What's done
- **HNOCA paper read.** 1.7M cells, 36 datasets, 26 protocols (3 unguided + 23 guided),
  day 7 → day 450, harmonised cell-type labels and primary-reference similarity scores.
  Data on Zenodo (records `14160929` full / `14161275` cleaned), code on
  `theislab/neural_organoid_atlas`, toolbox on `devsystemslab/HNOCA-tools`.
  Full notes in `docs/hnoca_summary.md`.
- **5 candidate tasks defined** with objective / input / output / metrics / baselines:
  T1 next-timepoint pseudobulk, T2 out-of-protocol cell-type, T3 organoid→primary
  fidelity, T4 perturbation/condition response, T5 real-time pseudotime. Full spec in
  `docs/tasks.md`.
- **MVP repo scaffolded** (`README.md`, `pyproject.toml`, `src/{data,tasks,models,eval,plot}/`,
  `configs/`, `tests/`, `notebooks/`).
- **Sandbox smoke test** runs T1, T2, T5 on synthetic HNOCA-shaped data with three
  baselines each. 4/4 pytest tests pass.
- **9 plots** generated to `plots/`.

## Sandbox results (synthetic data, sanity check only)
| Task | Model | Key metric | Value |
|------|-------|-----------|-------|
| T1 next-timepoint | identity (`x_t = x_{t+1}`) | Pearson | 0.974 |
| T1 next-timepoint | population mean | Pearson | 0.807 |
| T1 next-timepoint | linear (Ridge) | Pearson | **0.995** |
| T2 cell-type OOP | logreg | macro-F1 | 0.385 |
| T2 cell-type OOP | KNN (k=15) | macro-F1 | 0.386 |
| T5 pseudotime | PC1 | Spearman | 0.874 |

The numbers above are *not* HNOCA results — they're a smoke test that the harness,
metrics, and plotting work. Linear beats identity (good — model has real signal),
identity beats population mean by a wide margin (good — `x_t` carries information),
and PC1 captures most of the synthetic time signal.

## What week 2 looks like
1. Pull HNOCA cleaned (Zenodo `14161275`) on a machine with network and
   subset to telencephalon + 4 protocols.
2. Re-run T1 / T2 / T5 with the same harness on real data — same code,
   real `adata`.
3. Add T3 (fidelity score) as soon as the primary-reference similarity column
   is in `adata.obs`.
4. Wire one foundation model (Geneformer or scGPT) behind `BaselineModel.predict`
   and add it to the leaderboard.
5. Project Paola's organoids into the HNOCA latent (HNOCA-tools `map_query`)
   and flag (i) unmatched cell populations, (ii) trajectory deviations from the
   closest atlas protocol.

## Repo layout
```
organoid-agent-mvp/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/                # YAML configs per task
├── docs/                   # paper summary, task spec, decisions, this file
├── notebooks/01_explore.py # generates plots/ from synthetic data
├── plots/                  # PNGs + results.csv from 01_explore.py
├── src/
│   ├── data/{hnoca.py, synthetic.py}
│   ├── tasks/{base.py, t1_next_timepoint.py, t2_celltype_oop.py, t5_pseudotime.py}
│   ├── models/{base.py, baselines.py}
│   ├── eval/{metrics.py, runner.py}
│   └── plot/figures.py
└── tests/test_smoke.py
```

## Things to discuss with the lab
- Lock the first **region** (telencephalon is biggest but most heterogeneous).
- Decide between **time-course** vs **snapshot** organoids on Paola's side —
  T1 / T5 require time-course; T2 / T3 work on snapshots.
- Pick the first **foundation model** (Geneformer / scGPT / scFoundation / UCE).
- T4 (perturbation) needs a clear "control vs treated" mapping in HNOCA's
  metadata — what's the cleanest split?
