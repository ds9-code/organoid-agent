# Convenience targets.
#
# Quick start:
#     make data        # streamed ~80 MB HNOCA subset from Zenodo
#     make explore     # printed summary of what's in the subset
#     make plots       # regenerate the 9 PNG plots + leaderboards
#     make demo        # ask the agent one example question (needs OPENAI_API_KEY)
#     make agent       # interactive agent REPL

PYTHON ?= python
DEMO_QUESTION ?= What cells dominate a Velasco day-100 cortical organoid?

.PHONY: help data explore plots demo agent test eval eval-no-tools build-questions \
        eval-tasks eval-t02 eval-t03 eval-t08 eval-t10 eval-t11

help:
	@echo "Targets:"
	@echo "  make data      - stream a small real-HNOCA subset (~80 MB) from Zenodo"
	@echo "  make explore   - print a sectioned summary of data/hnoca_dt_subset.h5ad"
	@echo "  make plots     - regenerate plots/*.png from the real subset"
	@echo "  make agent     - start the interactive agent REPL (needs OPENAI_API_KEY)"
	@echo "  make demo      - one-shot agent: $(DEMO_QUESTION)"
	@echo "  make eval-tasks- run all five per-task scientific evals (T2,T3,T8,T10,T11)"
	@echo "  make eval-t02  - just the cell-type annotation eval"
	@echo "  make test      - run the pytest smoke test"

data:
	$(PYTHON) download_data.py

explore:
	$(PYTHON) explore_hnoca.py

plots:
	PYTHONPATH=. $(PYTHON) notebooks/01_explore.py

agent:
	PYTHONPATH=. $(PYTHON) -m organoid_agent

demo:
	PYTHONPATH=. $(PYTHON) -m organoid_agent "$(DEMO_QUESTION)"

demo-plan:
	PYTHONPATH=. $(PYTHON) -m organoid_agent --cite "$(DEMO_QUESTION)"

test:
	$(PYTHON) -m pytest tests/ -q

build-questions:
	PYTHONPATH=. $(PYTHON) benchmarks/build_atlas_recall.py

eval:
	PYTHONPATH=. $(PYTHON) -m benchmarks.eval --mode agent

eval-no-tools:
	PYTHONPATH=. $(PYTHON) -m benchmarks.eval --mode no_tools

eval-with-plan:
	PYTHONPATH=. $(PYTHON) -m benchmarks.eval --mode agent_plan

# ---------------------------------------------------------------------------
# Per-task scientific evaluations (the spreadsheet tasks).
# Each is a self-contained Python script — no LLM, just predictions + metrics.
# ---------------------------------------------------------------------------
eval-tasks: eval-t02 eval-t08 eval-t11 eval-t03 eval-t10

eval-t02:
	@echo "=== T2 Cell-Type Annotation (coarse) ==="
	PYTHONPATH=. $(PYTHON) -m benchmarks.tasks.task_02_cell_type_annotation --granularity coarse
	@echo
	@echo "=== T2 Cell-Type Annotation (fine) ==="
	PYTHONPATH=. $(PYTHON) -m benchmarks.tasks.task_02_cell_type_annotation --granularity fine

eval-t08:
	@echo "=== T8 Out-of-Protocol Generalisation (coarse) ==="
	PYTHONPATH=. $(PYTHON) -m benchmarks.tasks.task_08_oop_generalization --granularity coarse
	@echo
	@echo "=== T8 Out-of-Protocol Generalisation (fine) ==="
	PYTHONPATH=. $(PYTHON) -m benchmarks.tasks.task_08_oop_generalization --granularity fine

eval-t11:
	@echo "=== T11 Novel-Cell Detection ==="
	PYTHONPATH=. $(PYTHON) -m benchmarks.tasks.task_11_novel_cell_detection

eval-t03:
	@echo "=== T3 Composition Shift Over Time ==="
	PYTHONPATH=. $(PYTHON) -m benchmarks.tasks.task_03_composition_shift

eval-t10:
	@echo "=== T10 Cross-Protocol Composition Transfer ==="
	PYTHONPATH=. $(PYTHON) -m benchmarks.tasks.task_10_cross_protocol_transfer
