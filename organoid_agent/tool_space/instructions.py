"""
Tool registry — defines the OpenAI-compatible function-calling schema for every
tool the agent can call.

Mirrors ``medea/tool_space/instructions.py`` + ``medea_tools_config.json``: the
registry is declarative (a dict) so the Analysis module can emit the JSON-schema
list without having to know about each tool's signature.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TOOL_CONFIG_PATH = Path(__file__).resolve().parent / "tool_config.json"


def _load_config() -> list[dict[str, Any]]:
    with _TOOL_CONFIG_PATH.open() as f:
        return json.load(f)["tools"]


TOOL_REGISTRY: list[dict[str, Any]] = _load_config()


def get_openai_tool_schemas(include_pubmed: bool = True) -> list[dict[str, Any]]:
    """Return the list of OpenAI function-calling schemas the LLM will see.

    The PubMed tools are optional — set ``include_pubmed=False`` to evaluate
    HNOCA-only agents (e.g., in the citation-grounding ablation).
    """
    out: list[dict[str, Any]] = []
    for t in TOOL_REGISTRY:
        if not include_pubmed and t.get("tool_module") == "pubmed":
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        })
    return out


def list_tools() -> list[str]:
    return [t["name"] for t in TOOL_REGISTRY]
