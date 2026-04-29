# Decisions / open questions

## Decided
- Anndata as the canonical format. Everything goes through `scanpy`.
- Tasks share a `(adata, split) -> (X, y)` interface so the eval harness is
  task-agnostic.
- Baselines must include at least one *trivial* baseline (identity / class
  prior) per task so we can detect cheating.
- All tasks emit a single `Dict[str, float]` so the leaderboard is uniform.

## Open
- Which foundation models to wire up first? Candidates: Geneformer, scGPT,
  scFoundation, UCE.
- Single-region vs. all-HNOCA. Single region is faster to iterate on.
- Where do Paola's organoids land — same `Dataset` interface, or a sibling
  loader with extra metadata?
