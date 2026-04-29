# HNOCA — Summary for the MVP

**Paper.** He Z., Dony L., Fleck J. S., et al. *An integrated transcriptomic cell
atlas of human neural organoids.* Nature, 2024. DOI: 10.1038/s41586-024-08172-8.
Reproducibility code: `theislab/neural_organoid_atlas`. Toolbox:
`devsystemslab/HNOCA-tools`. Cleaned data: Zenodo `14161275`. Full data:
Zenodo `14160929`.

## What it is
An integrated single-cell transcriptomic atlas of **human neural organoids**:
- **~1.7M cells** drawn from **36 datasets** generated with **26 differentiation
  protocols** (3 unguided, 23 guided).
- Time points span **day 7 → day 450**, giving a real time axis for
  trajectory / dynamics models.
- Built and analysed with **scanpy 1.9.3**; integration with scVI/scANVI;
  pseudotime that is informed by real time via **moscot** neural OT.
- Mapped to a curated **primary developing-brain** reference so each organoid
  cell can be scored for fidelity to its primary counterpart.

## Cell coverage by region (approximate, from the paper)
| Region                | ~Cells   |
|-----------------------|----------|
| Telencephalon         | 802,509  |
| Cerebral cortex       | 353,984  |
| Brain (unspecified)   | 287,461  |
| Midbrain tegmentum    | 156,397  |
| Other (diencephalon, hindbrain, retinal, etc.) | balance |

Each cell has annotations for protocol, age (days in vitro), region label,
cell-type call, and a similarity score to the primary reference.

## Why it's a good benchmark substrate
- **Real time labels** → temporal prediction tasks (t → t+Δ) are well-posed.
- **Many protocols** → out-of-protocol generalisation is testable
  (train on protocol set A, evaluate on protocol B).
- **Primary reference** → fidelity-style metrics beyond reconstruction
  (e.g., does the predicted future look more like the matched primary state?).
- **Region heterogeneity** → models that exploit region context vs. agnostic
  baselines can be separated.

## Practical access
- Zenodo h5ad files are large; recommended to subset on disk
  (e.g., a single region or a single protocol family) before iterating.
- HNOCA-tools provides a one-call mapper that annotates new query cells
  against the atlas — useful as a strong "retrieval" baseline.

## What we'll do for Paola's organoids
The atlas's `map_query` style annotation lets us project Paola's cells into
HNOCA's latent space and look for **(a)** unmatched populations (candidate
novel cell states) and **(b)** trajectory deviations from the closest atlas
protocol — both are usable as discovery signals.
