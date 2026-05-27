"""
organoid_agent — Medea-style multi-module AI agent for brain-organoid biology.

Three collaborating modules, mirroring Medea's ResearchPlanning / Analysis /
LiteratureReasoning, specialised for scRNA-seq + HNOCA workflows.  Backed by
any OpenAI-compatible LLM endpoint (default: Hermes-4 via Nous Portal).
"""
from __future__ import annotations

__version__ = "0.1.0"

# Core workflow functions
from .core import organoid_agent, experiment_analysis, literature_reasoning  # noqa: F401

# Module classes
from .modules.agent_llms import AgentLLM, LLMConfig  # noqa: F401
from .modules.research_planning import ResearchPlanning  # noqa: F401
from .modules.analysis import Analysis  # noqa: F401
from .modules.literature_reasoning import LiteratureReasoning  # noqa: F401

# Tool classes (HNOCA is the headline tool; PubMed is the literature tool)
from .tool_space.hnoca import HNOCAModel  # noqa: F401
from .tool_space.pubmed import PubMedTool  # noqa: F401

__all__ = [
    "organoid_agent",
    "experiment_analysis",
    "literature_reasoning",
    "AgentLLM",
    "LLMConfig",
    "ResearchPlanning",
    "Analysis",
    "LiteratureReasoning",
    "HNOCAModel",
    "PubMedTool",
]
