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

.PHONY: help data explore plots demo agent test

help:
	@echo "Targets:"
	@echo "  make data      - stream a small real-HNOCA subset (~80 MB) from Zenodo"
	@echo "  make explore   - print a sectioned summary of data/hnoca_dt_subset.h5ad"
	@echo "  make plots     - regenerate plots/*.png from the real subset"
	@echo "  make agent     - start the interactive agent REPL (needs OPENAI_API_KEY)"
	@echo "  make demo      - one-shot agent: $(DEMO_QUESTION)"
	@echo "  make test      - run the pytest smoke test"

data:
	$(PYTHON) download_data.py

explore:
	$(PYTHON) explore_hnoca.py

plots:
	PYTHONPATH=. $(PYTHON) notebooks/01_explore.py

agent:
	$(PYTHON) -m agent.agent

demo:
	$(PYTHON) -m agent.agent "$(DEMO_QUESTION)"

test:
	$(PYTHON) -m pytest tests/ -q
