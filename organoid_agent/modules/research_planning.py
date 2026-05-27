"""
Research-Planning module — the first stage of a Medea-style 3-module agent.

Mirrors ``medea.modules.research_planning``. Given a user question, asks an LLM
to (1) restate the question concretely, (2) identify which HNOCA tool category
should answer it, and (3) flag any biology context the analysis layer needs to
honour.

This module never calls tools itself; it produces a *plan* that the Analysis
module then executes.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

from .agent_llms import AgentLLM


_PLANNING_PROMPT = textwrap.dedent("""
    You are the research-planning module of a brain-organoid AI agent.

    Available tool categories (in the analysis module):
      - atlas_query : composition / gene-expression / kNN queries on HNOCA
      - classifier : HNOCA-trained cell-type classifier (logreg or kNN)
      - prediction : kNN retrieval to predict an unseen (protocol, age) state
      - literature : PubMed / paper search

    Given the USER QUESTION below, output a JSON object with these keys:
      restated   : a concrete restatement of the question (resolve loose
                   phrases like "high CHIR" or "around 3 months" to concrete
                   values where reasonable)
      tool_category : which of the four categories above is the primary fit
      tools         : ordered list of specific tool names you would call
                      (e.g. ["query_composition", "gene_expression_timecourse"])
      caveats       : list of facts the analysis module must honour
                      (e.g. "HNOCA subset is dorsal telencephalon only;
                       reject queries about midbrain")
      bench_relevant : true/false — is this an HNOCA-answerable factual
                       question, or pure biology background?

    Output ONLY the JSON. No prose.

    USER QUESTION:
    {question}
""").strip()


@dataclass
class ResearchPlan:
    restated: str
    tool_category: str
    tools: list[str]
    caveats: list[str]
    bench_relevant: bool
    raw: str

    def as_text(self) -> str:
        """Human-readable summary that the Analysis module's system prompt can ingest."""
        return (
            f"Restated question: {self.restated}\n"
            f"Primary tool category: {self.tool_category}\n"
            f"Suggested tools: {', '.join(self.tools) if self.tools else '(none)'}\n"
            f"Caveats: {'; '.join(self.caveats) if self.caveats else '(none)'}\n"
        )


class ResearchPlanning:
    """First module of the agent. Produces a plan; does not execute tools."""

    def __init__(self, llm: AgentLLM, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose

    def __call__(self, question: str) -> ResearchPlan:
        messages = [
            {"role": "system", "content": "You output strict JSON only — no prose."},
            {"role": "user", "content": _PLANNING_PROMPT.format(question=question)},
        ]
        resp = self.llm.chat(messages)
        raw = resp.choices[0].message.content or "{}"
        # Strip code fences if present
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            obj = json.loads(clean)
        except json.JSONDecodeError:
            obj = {}
        plan = ResearchPlan(
            restated=str(obj.get("restated") or question),
            tool_category=str(obj.get("tool_category") or "atlas_query"),
            tools=list(obj.get("tools") or []),
            caveats=list(obj.get("caveats") or []),
            bench_relevant=bool(obj.get("bench_relevant", True)),
            raw=raw,
        )
        if self.verbose:
            print(f"[ResearchPlanning] tool_category={plan.tool_category} "
                  f"tools={plan.tools}")
        return plan
