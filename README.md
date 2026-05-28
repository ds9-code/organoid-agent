# Organoid Agent

**The goal:** build an LLM-powered agent that can do science on brain
organoids — answer "what would my organoid look like?" and "is this
patient organoid abnormal?" type questions by calling tools, not by
guessing from training data — and **benchmark it on a defined set of
prediction tasks** so we can show it beats frontier LLMs before moving
to private Arlotta-Lab data.

Modelled on [**Medea**](https://github.com/mims-harvard/medea) (Sui, Li,
…, Zitnik, bioRxiv 2026). Powered by **Hermes-4** running locally via
Ollama — no API keys, no recurring cost. Drop-in compatible with any
OpenAI-style endpoint (OpenAI, OpenRouter, Nous Portal, Azure) via
two env-vars.

```
hermes -z "What cells dominate a Velasco day-100 cortical organoid?"

  -> calling mcp_organoid_agent_query_composition({
       "protocol": "Velasco", "age_min": 80, "age_max": 120
     })

~91% dorsal-telencephalic neurons (real HNOCA cells, n=889 across 42
organoids in this window) — dominated by cerebral-cortex pyramidal
(43.3%), extratelencephalic-projecting glutamatergic cortical (23.1%),
and pyramidal (18.9%) sub-identities. Consistent with mid-stage
Velasco-protocol cortical organoids being mostly post-mitotic deep-
layer neurons.
```

---

## The benchmark — what we're actually evaluating

The agent has to be measured against something. Our task spec is in
[`docs/project_notes.md`](docs/project_notes.md) (and the lab's working
spreadsheet). 15 candidate tasks in total; 5 of them are evaluable today
on real HNOCA data alone.

| ID | Task | What we measure | Status |
|---|---|---|---|
| **T2** | Cell-type annotation from expression | macro-F1 (held-out cells) | ✅ implemented |
| **T3** | Cell-type composition shift over time (within protocol) | KL divergence on proportion vectors | ✅ implemented |
| **T8** | Out-of-protocol cell-type generalisation | macro-F1 (held-out protocols) | ✅ implemented |
| **T10** | Cross-protocol composition transfer (Velasco → Lancaster) | KL on composition | ✅ implemented |
| **T11** | Novel-cell detection (atlas mapping) | AUROC | ✅ implemented |
| T1, T4, T7 | Donor-growth / neurotoxicity / fate-potential (Chimeroid) | Spearman, KL | needs Chimeroid SCP2609 download |
| T5 | Transcriptional age | MAE + Pearson on age in days | needs lab-internal long-term data |
| T6 | Survival under media conditions | MAE on functional readouts | needs lab-internal APM/CDM4 data |
| T9 | Primary-reference fidelity | Spearman vs HNOCA's published score | needs Braun atlas |
| T12 | Stress-signature detection | AUROC | needs Bhaduri 2020 |
| T13 | Disease-class classification | macro-F1 | needs Gleeson NDD biobank |
| T14 | CRISPR-perturbation vulnerability | Spearman rank | needs CHOOSE GSE228882 |
| T15 | Morphogen → regional fate | KL divergence | needs Amin/Pasca GSE269308 |

### Run the benchmark

```bash
make eval-tasks            # runs T2, T3, T8, T10, T11
make eval-t02              # just T2
make eval-t11              # just T11   (and so on)
```

Each writes a JSON of per-predictor metrics to `benchmarks/results/`
and prints a markdown table to stdout. **No LLM in the loop for these**
— they're pure-Python ML evaluations of the tools, because that's the
right shape for measuring predictive performance independently of which
agent layer is on top.

### First-run baseline numbers (logreg + kNN on PCA-30, no foundation model yet)

| Task | Best baseline | Score | What it tells us |
|---|---|---:|---|
| T2 (coarse, 3-class) | logreg | macro-F1 = **0.90** | Already saturated. |
| T2 (fine, 28-class) | logreg | macro-F1 = **0.47** | Real headroom for a foundation model. |
| T8 (fine, OOP) | logreg | macro-F1 = **0.12** | **0.35 drop vs T2** — classifier learns protocol-specific quirks, not biology. |
| T11 | kNN-distance k=5 | AUROC = **0.88** | Hits ≥ 0.85 target from the project notes. |
| T3 | pop-mean | mean KL = **0.16** | Identity baseline at 0.72 — composition really does change with age. |
| T10 (Velasco→Lancaster) | age-matched target mean | mean KL = **0.09** | Clear headroom for CellFlow / FM-conditioned predictors. |

These are what a foundation-model wrapper (Geneformer / scGPT / UCE)
has to beat to be worth shipping.

---

## The agent

For interactive use ("ask in English, get an answer grounded in real
HNOCA cells") the same tools are exposed via:

1. **A Python orchestration package** (`organoid_agent/`) modelled on
   Medea's 3-module design.
2. **An MCP server** (`organoid_agent/mcp_server.py`) so Hermes-CLI /
   Claude Desktop / Cursor / Zed can call the tools natively.

Both share the same `tool_space/` (HNOCAModel + PubMedTool) and the
same LLM backbone (Hermes-4-14B by default, swappable). The benchmark
numbers above don't depend on which agent layer is on top — they
measure the tools.

### Architecture

| Module | File | Role |
|---|---|---|
| **Research Planning** | [`organoid_agent/modules/research_planning.py`](organoid_agent/modules/research_planning.py) | Restates the question, picks a tool category, surfaces caveats. No tools called. |
| **Analysis** | [`organoid_agent/modules/analysis.py`](organoid_agent/modules/analysis.py) | Tool-calling chat loop. Narrates biologically grounded answers. |
| **Literature Reasoning** | [`organoid_agent/modules/literature_reasoning.py`](organoid_agent/modules/literature_reasoning.py) | PubMed-grounded citation pulling per claim. |
| **LLM client** | [`organoid_agent/modules/agent_llms.py`](organoid_agent/modules/agent_llms.py) | Pluggable OpenAI-compatible client. Default: local Hermes-4-14B via Ollama. |
| **MCP server** | [`organoid_agent/mcp_server.py`](organoid_agent/mcp_server.py) | Exposes the tool space to Hermes-CLI and any other MCP client. |

### Tool space

| Tool | Used by which task | What it does |
|---|---|---|
| `query_composition` | T2 (agent demo), T11 (agent demo) | Composition of cells matching a protocol / age filter |
| `gene_expression_timecourse` | (atlas-recall demos) | Mean log1p expression of one gene by age bin |
| `predict_composition_at_age` | T3, T10 | kNN-retrieved composition at nearest available ages |
| `find_similar_cells` | T11 | Latent-space neighbours — which atlas cells this query sits near |
| `classify_cell_type` | T2, T8 | HNOCA-trained logreg / kNN classifier |
| `pubmed_search` | citation grounding (future) | NCBI PubMed search |
| `fetch_abstract` | citation grounding (future) | NCBI abstract fetch by PMID |

All tools answer from real data — never the LLM's parametric memory.

---

## Setup

```bash
# 1. Python env
conda create -n organoid-agent python=3.11 -y
conda activate organoid-agent
pip install -r requirements.txt

# 2. Local LLM backbone — Hermes-4-14B via Ollama (~9 GB, free)
brew install ollama
brew services start ollama
ollama pull hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M

# 3. Real HNOCA subset (~80 MB; streamed from Zenodo via HTTP range reads)
make data
```

For a different backbone (Nous Portal / OpenAI / OpenRouter), see the
alternative blocks in `.env.example`.

## Run the benchmark

```bash
make eval-tasks            # all 5 implemented per-task evals (T2 T3 T8 T10 T11)
make eval-t02              # just T2 (cell-type annotation, held-out)
make eval-t11              # just T11 (novel-cell detection, AUROC)
make eval-t08              # just T8 (out-of-protocol generalisation)
make eval-t03              # just T3 (composition shift over time)
make eval-t10              # just T10 (cross-protocol transfer)
```

These are the **task evaluations** the project is being measured by.
Pure Python. No LLM call. Each writes timestamped JSON to
`benchmarks/results/` (gitignored).

## Run the agent (Python)

```bash
make agent       # interactive REPL — 3-module pipeline
make demo        # one-shot question
make eval        # the 11-question atlas-recall LLM-graded benchmark
make eval-no-tools   # frontier-LLM baseline (same LLM, no tool calls)
```

## Run the agent (Hermes-CLI via MCP)

One-time registration in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  organoid_agent:
    command: "/opt/anaconda3/bin/python3"     # or your Python path
    args: ["-m", "organoid_agent.mcp_server"]
    env:
      PYTHONPATH: "/Users/<you>/organoid-agent"
    timeout: 120
```

Verify and use:

```bash
hermes mcp list                       # should show 'organoid_agent ✓ enabled'
hermes mcp test organoid_agent        # 7 tools discovered
hermes -z "What cells dominate Velasco day-100 cortical organoids?"
```

## Use as a library (Python)

```python
from organoid_agent import (
    AgentLLM, LLMConfig, HNOCAModel, PubMedTool,
    ResearchPlanning, Analysis, LiteratureReasoning,
    organoid_agent,
)

llm = AgentLLM(LLMConfig(temperature=0.4))
hnoca = HNOCAModel(); pubmed = PubMedTool()
analysis = Analysis(llm, hnoca, pubmed=pubmed)
result = organoid_agent(
    "Do ARID1B-mutant cortical organoids show OPC expansion?",
    ResearchPlanning(llm), analysis, LiteratureReasoning(llm, pubmed),
    do_cite=True,
)
print(result.final)
```

---

## Layout

```
organoid_agent/                     # the Python package (Medea-style)
├── __init__.py
├── __main__.py                     # python -m organoid_agent — REPL or one-shot
├── core.py                         # 3-module workflows
├── modules/
│   ├── agent_llms.py               # AgentLLM, LLMConfig
│   ├── research_planning.py        # ResearchPlanning module
│   ├── analysis.py                 # Analysis module (tool-calling loop)
│   └── literature_reasoning.py     # LiteratureReasoning module
├── tool_space/
│   ├── hnoca.py                    # HNOCAModel — 5 tools over the HNOCA atlas
│   ├── pubmed.py                   # PubMedTool — search + fetch via NCBI
│   ├── instructions.py             # OpenAI tool-schema builder
│   └── tool_config.json            # declarative tool registry (single source of truth)
└── mcp_server.py                   # stdio MCP server for Hermes-CLI etc.

benchmarks/
├── tasks/                          # ★ THE TASK EVALUATIONS ★
│   ├── common.py                   # shared helpers (split, KL, F1, AUROC)
│   ├── task_02_cell_type_annotation.py
│   ├── task_03_composition_shift.py
│   ├── task_08_oop_generalization.py
│   ├── task_10_cross_protocol_transfer.py
│   └── task_11_novel_cell_detection.py
├── build_atlas_recall.py           # rebuilds ground truth for atlas-recall demo eval
├── eval.py                         # the agent vs no-tools demo eval (LLM-graded)
└── questions/atlas_recall.yaml     # 11 atlas-recall questions

scripts/
└── download_hnoca_subset.py        # streaming HTTP-range subset downloader

data/                               # downloaded subsets (gitignored)
docs/project_notes.md (+ .pdf)      # full task spec, lit review, pivots
demo/example_session.md             # annotated agent transcript
```

## What lives where

- **This GitHub repo:** code only.
- **Zenodo** ([record 14161275](https://zenodo.org/records/14161275)): the HNOCA atlas. Streamed on demand, never committed.
- **Local Ollama:** the Hermes-4-14B model weights (~9 GB on disk).

## Roadmap

- [x] Real HNOCA streamed; plots from real cells, not synthetic
- [x] 7 tools wired (5 HNOCA + 2 PubMed)
- [x] Medea-style 3-module Python agent
- [x] Hermes-4-14B local backbone via Ollama (free)
- [x] MCP server for Hermes-CLI integration
- [x] **Per-task scientific evals for T2, T3, T8, T10, T11** (real numbers above)
- [x] Atlas-recall LLM-graded benchmark (11 questions)
- [ ] Plug a foundation-model embedding (Geneformer / scGPT / UCE) → re-run T2/T8/T11 → quantify the gap
- [ ] Bootstrap confidence intervals for T3 / T10 (low-sample-size tasks)
- [ ] Download Bhaduri / CHOOSE / Gleeson / Amin-Pasca → implement T12 / T14 / T13 / T15
- [ ] Citation-grounding evaluation (real-PMID audit)
- [ ] CellFlow integration for T10 / T15
- [ ] Project Paola's organoids into the HNOCA latent (the eventual Arlotta use case)
