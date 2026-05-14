"""
Conversational agent for the HNOCA organoid screen (OpenAI SDK).

The LLM does the reasoning layer: it interprets a natural-language request
about brain-organoid protocols / ages / cell types, calls the
:class:`HNOCAModel` tool (which answers from real HNOCA cells), and explains
the result biologically.  The model layer never invents numbers -- every
quantity comes from the real subset in ``data/hnoca_dt_subset.h5ad``.

Uses the **OpenAI Python SDK**, so it works with the official OpenAI API or
any OpenAI-compatible endpoint (Dartmouth, vLLM/LiteLLM proxies, ...) --
point ``OPENAI_BASE_URL`` at it.

Usage
-----
    # 1. put OPENAI_API_KEY (and optionally OPENAI_BASE_URL / AGENT_MODEL)
    #    in a .env file (see .env.example)
    # 2. make sure data/hnoca_dt_subset.h5ad exists (python download_data.py)

    python -m agent.agent                                        # interactive
    python -m agent.agent "What cells dominate a Velasco day-100 organoid?"

Environment variables
---------------------
    OPENAI_API_KEY      required  (e.g. "sk-...")
    OPENAI_BASE_URL     optional  (default: official OpenAI API)
    AGENT_MODEL         optional  (default "gpt-4o")
    HNOCA_DATA_PATH     optional  (default "<repo>/data/hnoca_dt_subset.h5ad")
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import traceback
from pathlib import Path

from agent.hnoca_model import DEFAULT_DATA, HNOCAModel

DEFAULT_AGENT_MODEL = "gpt-4o"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_composition",
            "description": (
                "Return the cell-type composition of REAL HNOCA cells matching a "
                "filter on protocol and/or age range. Use this when the user asks "
                "'what cells do I see in protocol X at age Y?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "protocol": {"type": "string", "description": "Protocol short name, e.g. 'Velasco'"},
                    "age_min": {"type": "number", "description": "Minimum organoid age in days"},
                    "age_max": {"type": "number", "description": "Maximum organoid age in days"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gene_expression_timecourse",
            "description": (
                "Mean log1p expression of one gene across age bins, optionally "
                "restricted to a single protocol. Use for 'how does gene X change "
                "with organoid age?' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gene": {"type": "string", "description": "Exact gene symbol, e.g. 'NEUROD6'"},
                    "protocol": {"type": "string", "description": "Optional protocol short name"},
                },
                "required": ["gene"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_composition_at_age",
            "description": (
                "Predict the cell-type composition for a (protocol, target age) pair "
                "by kNN retrieval over real HNOCA cells at the nearest available "
                "ages. Use this for 'what should my Velasco day-100 organoid look like?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "protocol": {"type": "string", "description": "Protocol short name"},
                    "target_age_days": {"type": "number", "description": "Target organoid age in days"},
                    "k_neighbors": {"type": "integer", "description": "How many nearby cells to pool"},
                },
                "required": ["protocol", "target_age_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_cells",
            "description": (
                "Given a (protocol, age) query, return what dominates its k nearest "
                "neighbours in the atlas's 30-D PCA latent space -- which protocols "
                "and cell types it sits near. The analogue of 'projecting a sample "
                "into the atlas to see what it looks like'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "protocol": {"type": "string"},
                    "age_days": {"type": "number"},
                    "k": {"type": "integer"},
                },
                "required": ["protocol", "age_days"],
            },
        },
    },
]


def build_system_prompt(model: HNOCAModel) -> str:
    spec = json.dumps(model.describe_inputs(), indent=2)
    return textwrap.dedent(
        f"""
        You are the interface to a *real* slice of the Human Neural Organoid Cell
        Atlas (HNOCA, He et al. Nature 2024) -- specifically a Dorsal-telencephalon
        subset streamed from Zenodo.  Every numeric answer you give MUST come from
        a tool call.  Do not invent compositions, gene expression values, or
        cell-type fractions.

        Tool input reference (valid protocols, age range, available cell types and
        the baseline composition):

        {spec}

        How to behave:
        - If the user describes a request loosely ("a mid-stage cortical organoid",
          "a Velasco organoid around 3 months"), map it to a concrete
          (protocol, age) and state the mapping you chose so the user can correct
          you.
        - Always call a tool before giving numbers. If a request needs several
          comparisons (e.g. "compare protocol A vs B"), call the tool more than
          once and contrast the results.
        - After a tool returns, give the headline composition or trend
          quantitatively, then a brief biological interpretation (e.g. "mostly
          radial glia at day 30 -> mostly cortical pyramidal neurons by day 100,
          consistent with the cortical neurogenesis timeline").
        - Be honest about caveats: the subset is ~4.4k cells / 16 protocols, only
          Dorsal telencephalon; many protocols only span part of the day range;
          extrapolated ages are flagged in the tool output -- pass that on.
        - Don't claim a gene exists in the atlas unless a tool confirms it.
        """
    ).strip()


def run_tool(model: HNOCAModel, name: str, tool_input: dict) -> dict:
    fn = getattr(model, name, None)
    if fn is None:
        return {"error": f"Unknown tool {name!r}."}
    try:
        return fn(**(tool_input or {}))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def chat(client, agent_model: str, model: HNOCAModel, messages: list) -> None:
    """Run one user turn to completion (handles any number of tool round-trips)."""
    while True:
        resp = client.chat.completions.create(
            model=agent_model, messages=messages, tools=TOOLS,
        )
        if not getattr(resp, "choices", None):
            raise RuntimeError(f"LLM endpoint returned no choices (model={agent_model!r}).")
        msg = resp.choices[0].message

        entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(entry)

        if msg.content:
            print(f"\nAgent: {msg.content.strip()}\n")

        if not msg.tool_calls:
            return

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  -> calling {tc.function.name}({json.dumps(args)})")
            result = run_tool(model, tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    except ImportError:
        pass  # dotenv is optional; env vars from the shell still work

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "Set it in your environment or create a .env file (see .env.example).",
            file=sys.stderr,
        )
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: the 'openai' package is not installed. Run: pip install openai",
              file=sys.stderr)
        return 1

    base_url = os.environ.get("OPENAI_BASE_URL") or None
    agent_model = os.environ.get("AGENT_MODEL", DEFAULT_AGENT_MODEL)
    data_path = os.environ.get("HNOCA_DATA_PATH", str(DEFAULT_DATA))

    print(f"Loading HNOCA tool from {data_path} ...\n")
    model = HNOCAModel(data_path=data_path, verbose=True)

    client = OpenAI(api_key=api_key, base_url=base_url)
    where = base_url or "https://api.openai.com/v1"
    print(f"LLM: model={agent_model!r} via {where}\n")

    system_prompt = build_system_prompt(model)
    messages: list = [{"role": "system", "content": system_prompt}]

    one_shot = " ".join(sys.argv[1:]).strip()
    if one_shot:
        messages.append({"role": "user", "content": one_shot})
        print(f"You: {one_shot}")
        chat(client, agent_model, model, messages)
        return 0

    print("=" * 70)
    print("HNOCA organoid agent. Ask about real cells in the Dorsal-telencephalon")
    print("subset of HNOCA (~4.4k cells, 16 protocols, days 15-300).")
    print("Examples:")
    print("  - What cells dominate a Velasco day-100 cortical organoid?")
    print("  - How does NEUROD6 expression change with organoid age?")
    print("  - Compare Velasco vs Lancaster organoids at day 60.")
    print("Type 'exit' or Ctrl-D to quit.")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in {"exit", "quit", ":q"}:
            break
        messages.append({"role": "user", "content": user_input})
        try:
            chat(client, agent_model, model, messages)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as exc:
            print(f"\n[agent error] {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
