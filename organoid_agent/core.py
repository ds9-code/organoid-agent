"""
Core entry points for organoid_agent — mirrors ``medea/core.py``.

Three top-level workflows:
  - ``organoid_agent(question, ...)`` :  full 3-module run (plan -> analyze -> cite)
  - ``experiment_analysis(question, ...)``: plan + analyze, no literature
  - ``literature_reasoning(question, ...)``: literature-only, no atlas

Each takes pre-constructed module instances so the same modules can be reused
across many questions without re-loading HNOCA every time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .modules.analysis import Analysis, AnalysisResult
from .modules.literature_reasoning import LiteratureReasoning, CitedClaim
from .modules.research_planning import ResearchPlanning, ResearchPlan


@dataclass
class OrganoidAgentResult:
    question: str
    plan: Optional[ResearchPlan]
    analysis: Optional[AnalysisResult]
    citations: list[CitedClaim] = field(default_factory=list)
    final: str = ""


def organoid_agent(
    question: str,
    planning_module: ResearchPlanning,
    analysis_module: Analysis,
    literature_module: Optional[LiteratureReasoning] = None,
    do_cite: bool = False,
) -> OrganoidAgentResult:
    """Full 3-module run: plan -> analyse -> (optionally) cite.

    The analysis module produces the final answer; the literature module is
    invoked only if ``do_cite=True`` AND the analysis result mentions claims
    worth grounding.
    """
    plan = planning_module(question)
    analysis = analysis_module.run(question, research_plan=plan.as_text())

    citations: list[CitedClaim] = []
    if do_cite and literature_module is not None and analysis.answer:
        # naive: cite the headline claim of the answer
        first_line = next(
            (l.strip() for l in analysis.answer.splitlines() if l.strip()), ""
        )
        if first_line:
            citations = literature_module.support_claim(first_line, k=3)

    return OrganoidAgentResult(
        question=question, plan=plan, analysis=analysis,
        citations=citations, final=analysis.answer,
    )


def experiment_analysis(
    question: str,
    planning_module: ResearchPlanning,
    analysis_module: Analysis,
) -> tuple[ResearchPlan, AnalysisResult]:
    """Plan + analyse without the literature module."""
    plan = planning_module(question)
    analysis = analysis_module.run(question, research_plan=plan.as_text())
    return plan, analysis


def literature_reasoning(
    claim: str,
    literature_module: LiteratureReasoning,
    k: int = 5,
) -> list[CitedClaim]:
    """Literature-only search for evidence supporting a claim."""
    return literature_module.support_claim(claim, k=k)
