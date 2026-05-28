# Organoid Agent

A multi-module AI agent for human-neural-organoid biology, modelled on
[**Medea**](https://github.com/mims-harvard/medea) (Sui, Li, …, Zitnik
bioRxiv 2026) and powered by **Hermes-4** running locally via Ollama
(no API keys, no recurring cost). Drop-in compatible with any OpenAI-style
endpoint — OpenAI, OpenRouter, Nous Portal, Azure — via two env-vars.

Three collaborating modules — Research Planning, Analysis, Literature
Reasoning — that orchestrate tool calls against the
[**Human Neural Organoid Cell Atlas**](https://www.nature.com/articles/s41586-024-08172-8)
(HNOCA, He et al. *Nature* 2024) and PubMed to answer organoid-biology
questions. Every quantitative claim is grounded in a tool return: the LLM
is never allowed to invent numbers, gene-expression values, or PMIDs.

```
You: What cells dominate a Velasco day-100 cortical organoid?

[ResearchPlanning] tool_category=atlas_query tools=['query_composition']
  -> calling query_composition({"protocol": "Velasco", "age_min": 80, "age_max": 120})

Agent: ~91% dorsal-telencephalic neurons (real HNOCA cells, n=889 across 42
       organoids in this window) — dominated by cerebral-cortex pyramidal
       (43.3%), extratelencephalic-projecting glutamatergic cortical
       (23.1%), and pyramidal (18.9%) sub-identities, with a residual ~4%
       radial-glia pool. Consistent with mid-stage Velasco-protocol
       cortical organoids being mostly post-mitotic deep-layer neurons.
```

See [`demo/example_session.md`](demo/example_session.md) for a longer
annotated transcript, or `make demo` once set up.

## Architecture

| Module | File | Role |
|---|---|---|
| **Research Planning** | [`organoid_agent/modules/research_planning.py`](organoid_agent/modules/research_planning.py) | Restates the question concretely; identifies which tool category should answer it; surfaces caveats. Produces a plan but does not call tools. |
| **Analysis** | [`organoid_agent/modules/analysis.py`](organoid_agent/modules/analysis.py) | The tool-calling chat loop. Calls HNOCA / PubMed tools, observes their returns, narrates a biologically grounded answer. |
| **Literature Reasoning** | [`organoid_agent/modules/literature_reasoning.py`](organoid_agent/modules/literature_reasoning.py) | Given a factual claim, finds PubMed papers that support it and pulls the single best supporting sentence from each abstract. Used for citation-grounding. |
| **LLM client** | [`organoid_agent/modules/agent_llms.py`](organoid_agent/modules/agent_llms.py) | Pluggable OpenAI-compatible chat client. Default: Hermes-4 via Nous Portal; override `OPENAI_BASE_URL` for any other endpoint. |

## Tool space

| Tool | What it does |
|---|---|
| `query_composition` | Cell-type composition of HNOCA cells matching a protocol / age filter |
| `gene_expression_timecourse` | Mean log1p expression of one gene by age bin |
| `predict_composition_at_age` | kNN-retrieved composition at the nearest available ages |
| `find_similar_cells` | Latent-space neighbours: which protocols + cell types this query sits near |
| `classify_cell_type` | HNOCA-trained logreg / kNN classifier (fine or coarse labels) |
| `pubmed_search` | NCBI E-utilities PubMed search |
| `fetch_abstract` | NCBI E-utilities abstract fetch by PMID |

All tools answer from real data — never the LLM's parametric memory.

## Setup

```bash
# 1. Python env
conda create -n organoid-agent python=3.11 -y
conda activate organoid-agent
pip install -r requirements.txt

# 2. Local LLM backbone — Hermes-4-14B via Ollama (free, ~9 GB on disk)
brew install ollama                                                                 # one-time
brew services start ollama                                                          # one-time
ollama pull hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M                   # ~9 GB
# (no API key needed; defaults in organoid_agent/modules/agent_llms.py point here)

# 3. Real HNOCA subset (~80 MB; streamed from Zenodo via HTTP range reads)
make data
```

For a different backbone — Nous Portal Hermes-4-405B, OpenAI GPT-4o,
OpenRouter, etc. — see the alternative blocks in `.env.example`.

There's a Makefile with shortcuts: `make data / explore / plots / agent /
demo / eval / eval-no-tools / eval-with-plan / test`.

## Run

```bash
make agent       # interactive REPL — the full 3-module pipeline
make demo        # one-shot question
make eval        # benchmark on 11 atlas-recall questions
```

The first call to `HNOCAModel(...)` takes ~10 s (loads subset, fits 30-D
PCA, trains classifiers, builds kNN); subsequent tool calls are sub-second.

## Benchmark

```bash
make build-questions      # rebuild ground truth from data/hnoca_dt_subset.h5ad
make eval                 # full 3-module agent
make eval-no-tools        # same LLM, no tool calls (frontier-LLM baseline)
make eval-with-plan       # research-planning + analysis (no literature)
```

Each run writes `benchmarks/results/<timestamp>_<mode>.csv` with per-question
pass/fail, tool calls used, the model's answer, and the grader's reasoning.
The headline score prints at the end.

The atlas-recall question set has 11 questions whose ground truth is
pre-computed from `data/hnoca_dt_subset.h5ad`. Frontier-LLM-no-tools should
score near zero (the answers depend on actually looking at the atlas, not on
text knowledge); the full agent should beat it by a wide margin.

## Using the package as a library

```python
from organoid_agent import (
    AgentLLM, LLMConfig,
    ResearchPlanning, Analysis, LiteratureReasoning,
    HNOCAModel, PubMedTool,
    organoid_agent, experiment_analysis, literature_reasoning,
)

llm = AgentLLM(LLMConfig(temperature=0.4))
hnoca = HNOCAModel()
pubmed = PubMedTool()

planning   = ResearchPlanning(llm)
analysis   = Analysis(llm, hnoca, pubmed=pubmed)
literature = LiteratureReasoning(llm, pubmed)

# Full 3-module run
result = organoid_agent(
    "Do ARID1B-mutant cortical organoids show OPC expansion?",
    planning, analysis, literature, do_cite=True,
)
print(result.final)
for c in result.citations:
    print(c.pmid, c.title, '--', c.evidence[:120])
```

## Layout

```
organoid_agent/                     # the Python package (Medea-style)
├── __init__.py
├── __main__.py                     # python -m organoid_agent — REPL or one-shot
├── core.py                         # organoid_agent() / experiment_analysis() / literature_reasoning()
├── modules/
│   ├── agent_llms.py               # AgentLLM, LLMConfig
│   ├── research_planning.py        # ResearchPlanning
│   ├── analysis.py                 # Analysis (the tool-calling chat loop)
│   └── literature_reasoning.py     # LiteratureReasoning
└── tool_space/
    ├── hnoca.py                    # HNOCAModel — 5 tools over the HNOCA atlas
    ├── pubmed.py                   # PubMedTool — search + fetch via NCBI
    ├── instructions.py             # tool registry + OpenAI schema builder
    └── tool_config.json            # declarative registry

benchmarks/
├── build_atlas_recall.py           # rebuilds ground truth from HNOCA
├── eval.py                         # runs agent on question set, grades, writes CSV
└── questions/atlas_recall.yaml     # 11 atlas-recall questions

scripts/
└── download_hnoca_subset.py        # streaming subset downloader (HTTP range reads)

data/                               # downloaded subsets (gitignored)
plots/                              # output figures (gitignored)
docs/                               # project notes + PDF
demo/                               # annotated agent transcript
```

## What lives where

- **This GitHub repo**: code only.
- **Zenodo** ([record 14161275](https://zenodo.org/records/14161275)): the HNOCA atlas. Streamed on demand, never committed.
- **Nous Portal**: the Hermes-4 LLM. API key only.

## Roadmap

- [x] Real HNOCA streamed; plots from real cells, not synthetic
- [x] 5 HNOCA tools wired
- [x] Medea-style 3-module architecture
- [x] Hermes-4 (Nous Portal) as the backbone LLM
- [x] PubMed tool for citation grounding
- [x] Atlas-recall benchmark with 11 questions
- [ ] Reproduction benchmark (the 18 brain-organoid papers in `docs/project_notes.md`)
- [ ] Citation-grounding evaluation (real-PMID audit)
- [ ] Foundation-model embedding tools (Geneformer, scGPT)
- [ ] CellFlow integration for perturbation prediction
- [ ] Project Paola's organoids into the HNOCA latent
