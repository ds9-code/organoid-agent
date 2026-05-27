"""
Interactive / one-shot CLI for the organoid_agent.

Usage:
    python -m organoid_agent                                # REPL
    python -m organoid_agent "What dominates Velasco D100?" # one-shot
    python -m organoid_agent --no-tools "..."               # no-tools baseline
    python -m organoid_agent --no-plan "..."                # skip planning module
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="One-shot question (omit for REPL)")
    ap.add_argument("--no-tools", action="store_true",
                    help="Disable function calling (frontier-LLM baseline)")
    ap.add_argument("--no-plan", action="store_true",
                    help="Skip the research-planning module")
    ap.add_argument("--cite", action="store_true",
                    help="Run literature module on the headline claim afterwards")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=True)
    except ImportError:
        pass

    from organoid_agent import (
        AgentLLM, LLMConfig,
        ResearchPlanning, Analysis, LiteratureReasoning,
        HNOCAModel, PubMedTool,
    )

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. cp .env.example .env and fill it in.",
              file=sys.stderr)
        return 1

    print("Loading modules...")
    llm = AgentLLM(LLMConfig(temperature=0.4), verbose=True)
    hnoca = HNOCAModel(verbose=True)
    pubmed = PubMedTool()
    planning = ResearchPlanning(llm, verbose=True)
    analysis = Analysis(llm, hnoca, pubmed=pubmed, verbose=True)
    literature = LiteratureReasoning(llm, pubmed, verbose=True)

    def run_one(q: str) -> None:
        plan = None if args.no_plan else planning(q)
        plan_text = plan.as_text() if plan else None
        result = analysis.run(q, research_plan=plan_text, use_tools=not args.no_tools)
        if args.cite:
            first_line = next((l for l in result.answer.splitlines() if l.strip()), "")
            if first_line:
                cites = literature.support_claim(first_line, k=3)
                print("\nCitations:")
                for c in cites:
                    print(f"  PMID {c.pmid}: {c.title}")
                    if c.evidence:
                        print(f"     evidence: {c.evidence[:200]}")

    one_shot = " ".join(args.question).strip()
    if one_shot:
        print(f"\nYou: {one_shot}")
        run_one(one_shot)
        return 0

    print("=" * 70)
    print("organoid_agent — type a question, 'exit' to quit.")
    print("=" * 70)
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit", ":q"}:
            break
        try:
            run_one(q)
        except Exception as exc:
            print(f"[error] {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
