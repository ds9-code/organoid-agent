"""
Analysis module — the "tools layer" of the three-module agent.

Mirrors ``medea.modules.experiment_analysis``. Given a user question (and
optionally a research plan from the Research-Planning module), runs an
LLM chat loop that can call tools registered in ``organoid_agent.tool_space``.
Every numeric claim must come from a tool call — the LLM is forbidden from
inventing.
"""
from __future__ import annotations

import json
import textwrap
import traceback
from dataclasses import dataclass
from typing import Any, Optional

from ..tool_space import get_openai_tool_schemas
from ..tool_space.hnoca import HNOCAModel
from ..tool_space.pubmed import PubMedTool
from .agent_llms import AgentLLM


@dataclass
class AnalysisResult:
    """Output of one analysis turn."""
    answer: str
    tool_calls: list[dict]
    messages: list[dict]
    error: Optional[str] = None


class Analysis:
    """Runs an LLM chat loop with tool calls. The middle module of the agent."""

    def __init__(
        self,
        llm: AgentLLM,
        hnoca: HNOCAModel,
        pubmed: PubMedTool | None = None,
        max_tool_rounds: int = 5,
        verbose: bool = True,
    ):
        self.llm = llm
        self.hnoca = hnoca
        self.pubmed = pubmed
        self.max_tool_rounds = max_tool_rounds
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # System prompt                                                      #
    # ------------------------------------------------------------------ #
    def build_system_prompt(self, research_plan: Optional[str] = None) -> str:
        spec = json.dumps(self.hnoca.describe_inputs(), indent=2)
        prompt = textwrap.dedent(
            f"""
            You are the analysis layer of a multi-module agent for human-neural-
            organoid biology. You answer questions about the HNOCA atlas
            (He et al. *Nature* 2024) by calling tools — never from parametric
            memory. Every numeric claim, composition fraction, gene-expression
            value or cell-type label must come from a tool return.

            Tool-input reference (valid protocols, ages, cell types, tools):

            {spec}

            How to behave:
            - Map loose requests ("a mid-stage Velasco organoid") to concrete
              (protocol, age) pairs and state the mapping.
            - For any quantitative answer, call the relevant tool first, then
              interpret the result biologically.
            - For literature questions, call the PubMed tool — do NOT invent
              author lists, journal names, or PMIDs.
            - Be honest about caveats: the HNOCA dorsal-telencephalon subset
              is ~4.4k cells, 16 protocols, days 15-300 — don't extrapolate
              outside that without flagging it.
            """
        ).strip()
        if research_plan:
            prompt += "\n\nResearch plan from the planning module:\n" + research_plan
        return prompt

    # ------------------------------------------------------------------ #
    # Tool dispatch                                                      #
    # ------------------------------------------------------------------ #
    def _run_tool(self, name: str, args: dict) -> dict:
        # HNOCA tools
        fn = getattr(self.hnoca, name, None)
        if fn is not None:
            try:
                return fn(**(args or {}))
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}
        # PubMed tools
        if self.pubmed is not None and hasattr(self.pubmed, name):
            try:
                return getattr(self.pubmed, name)(**(args or {}))
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}
        return {"error": f"unknown tool {name!r}"}

    # ------------------------------------------------------------------ #
    # Run                                                                #
    # ------------------------------------------------------------------ #
    def run(
        self,
        question: str,
        research_plan: Optional[str] = None,
        use_tools: bool = True,
    ) -> AnalysisResult:
        """Run one question to completion.

        If ``use_tools`` is False, runs the same LLM with no function-calling —
        the "frontier-LLM-no-tools" baseline used in benchmarking.
        """
        messages: list[dict] = [
            {"role": "system", "content": self.build_system_prompt(research_plan)},
            {"role": "user", "content": question},
        ]
        tools = get_openai_tool_schemas(include_pubmed=self.pubmed is not None) if use_tools else None
        tool_call_log: list[dict] = []
        answer = ""
        err: Optional[str] = None

        try:
            for _ in range(self.max_tool_rounds):
                resp = self.llm.chat(messages, tools=tools)
                msg = resp.choices[0].message

                entry: dict = {"role": "assistant", "content": msg.content or ""}
                if getattr(msg, "tool_calls", None):
                    entry["tool_calls"] = [
                        {
                            "id": tc.id, "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(entry)

                if msg.content:
                    answer = msg.content.strip()
                    if self.verbose:
                        print(f"\nAgent: {answer}\n")

                if not getattr(msg, "tool_calls", None):
                    break

                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    tool_call_log.append({"name": tc.function.name, "args": args})
                    if self.verbose:
                        print(f"  -> calling {tc.function.name}({json.dumps(args)})")
                    result = self._run_tool(tc.function.name, args)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str)[:4000],
                    })
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                traceback.print_exc()

        return AnalysisResult(answer=answer, tool_calls=tool_call_log,
                              messages=messages, error=err)
