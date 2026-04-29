# organoid-agent-mvp

MVP scaffold for an organoid-transcriptomics agent that benchmarks on HNOCA
(He, Dony, Fleck et al., *Nature*, 2024) and is intended to extend to Paola's
in-house organoid datasets.

## Status
- Paper read & summarised: see `docs/hnoca_summary.md`
- 5 candidate benchmark tasks drafted: see `docs/tasks.md`
- Repo skeleton: data loaders, task interface, evaluator, plotting, baselines
- Sandbox smoke test produces UMAP, time-course, and composition plots on
  synthetic HNOCA-shaped data (see `plots/` and `notebooks/01_explore.ipynb`-style
  script `notebooks/01_explore.py`).

## Layout
```
src/
  data/      AnnData loaders (HNOCA, Paola lab, synthetic)
  tasks/     Abstract Task + 5 concrete tasks
  models/    Baselines: identity, mean, KNN-in-latent, linear (placeholders for FM calls)
  eval/      Metric library (Pearson, MSE, accuracy, ARI, OT distance, FID-like)
  plot/      Plot helpers (UMAP, time-course, performance bars, calibration)
configs/     YAML configs for each task + dataset slice
notebooks/   Exploration & reporting
docs/        Paper summary, task spec, decisions log
plots/       Output figures
tests/       Pytest sanity checks
```

## How tasks are defined
Every task implements `src/tasks/base.Task` with:
- `prepare(adata) -> (X_train, y_train, X_eval, y_eval)`
- `predict(model, X_eval) -> y_pred`
- `evaluate(y_pred, y_eval) -> Dict[str, float]`

Models implement `src/models/base.BaselineModel` (`fit`, `predict`).
Foundation-model wrappers (Geneformer, scGPT, scFoundation, UCE, etc.) plug
into the same `BaselineModel` interface.



## Open decisions
Tracked in `docs/decisions.md`.
