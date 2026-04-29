# Benchmark Tasks (v0)

Five concrete tasks, ordered by how MVP-ready each is. Every task is a tuple
of (objective, input, output, metric, baseline, what-it-tests).

---

## T1. Next-timepoint transcriptome prediction (the "t → t+1" task)
**Objective.** Given the transcriptomic state of an organoid at day *t*,
predict its mean expression profile at day *t+Δ*.

**Input.** Pseudobulk expression vector `x_t ∈ R^{|G|}` for an
(organoid_id, region, protocol) at day *t*. Highly-variable genes only
(|G| ≈ 3,000).

**Output.** Pseudobulk vector `x_{t+Δ}` for the same organoid (or matched
organoid in held-out replicates).

**Metrics.**
- Pearson r over genes (per-sample, then averaged).
- MSE on log-normalised expression.
- ΔTF accuracy: fraction of curated lineage TFs whose direction-of-change
  sign is predicted correctly.

**Baselines.**
- *Identity* (`x_{t+Δ} = x_t`) — should be hard to beat at small Δ.
- *Population mean at t+Δ* across the protocol.
- *Linear* in scVI latent + decoder.
- *KNN in latent* — k=10 nearest organoids at t+Δ.
- *FM call* — Geneformer / scFoundation / scGPT zero-shot future-state.

**Why.** Cleanest formulation of "model dynamics", maps directly to the
project's stated objective.

---

## T2. Out-of-protocol cell-type prediction
**Objective.** Train a cell-type classifier on a subset of HNOCA protocols;
evaluate on held-out protocols.

**Input.** Single-cell expression vector.

**Output.** Cell-type label (HNOCA's harmonised ontology).

**Metrics.** Macro F1, per-class recall, ARI vs. ground-truth at the cluster
level, calibration (ECE).

**Baselines.** Logistic regression on HVGs, KNN in scVI latent, scANVI
transfer, HNOCA-tools `map_query` (very strong baseline), Geneformer linear
probe, scGPT zero-shot.

**Why.** Tests *generalisation across labs/protocols*, which is the failure
mode that matters for Paola's data.

---

## T3. Organoid → primary fidelity score
**Objective.** Predict, for each organoid cell, the similarity score to its
matched primary cell type (treating HNOCA's primary-reference distance as
ground truth).

**Input.** Single-cell expression + protocol + day.

**Output.** Scalar fidelity score in [0, 1].

**Metrics.** Spearman ρ vs. HNOCA's reported similarity, R², and
*ranking* AUC for "high vs. low fidelity" cells.

**Baselines.** Mean by protocol, scVI latent + linear regressor, transformer
features + linear probe.

**Why.** A model that nails fidelity is directly useful as a screening tool
for new protocols ("which protocol best mimics primary X?").

---

## T4. Perturbation/condition response prediction
**Objective.** For a set of organoid samples that share a condition label
(e.g., guided vs. unguided, with-vs-without a morphogen), predict the
expression change vector.

**Input.** Control sample mean + condition label.

**Output.** Treated sample mean.

**Metrics.** Pearson r on Δ-expression, top-K DEG overlap, sign accuracy.

**Baselines.** Mean shift from another protocol, CPA / scGen, FM-conditioned.

**Why.** Stress-tests *causal*-style prediction; sets up later in-silico
screen of Paola's perturbations.

---

## T5. Trajectory consistency (real-time-aware pseudotime)
**Objective.** Given a population of cells at one timepoint, infer the
ordering that best matches *real* day labels of held-out cells.

**Input.** Cell × gene matrix.

**Output.** Per-cell pseudotime in [0, 1].

**Metrics.** Spearman ρ vs. real day, Kendall τ, OT cost between predicted
and true day distributions (matches the moscot framing in the paper).

**Baselines.** Diffusion pseudotime, scVI latent + 1-D PHATE, moscot neural OT
(reference baseline from the paper itself).

**Why.** Probes whether the agent has internalised dynamics, not just
identity. Directly comparable to the paper's own trajectory analysis.

---

## Selection for week 1
We'll instrument **T1, T2, T5** end-to-end first (they share most of the
data pipeline), then add **T3** (one extra label column) and **T4** (needs a
condition split).

## Open questions for the lab
- Which *region* to lock first? Telencephalon has the most cells but also
  the most protocol heterogeneity.
- For Paola's data, do we have time-course or single-snapshot organoids?
  T1/T5 require time-course; T2/T3 work on snapshots.
- Foundation-model endpoint — local (Geneformer/scGPT weights) or hosted?
