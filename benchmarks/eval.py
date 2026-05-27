"""
Evaluation harness for the HNOCA organoid agent.

Loads a YAML question set, runs the agent on each question, captures tool
calls + final answer, grades against precomputed ground truth, writes a CSV
+ a printed summary.

Two modes:
    --mode agent       use HNOCAModel tools (default)
    --mode no_tools    same LLM, no tool calling -- baseline

Usage:
    cp .env.example .env  # fill OPENAI_API_KEY first
    python -m benchmarks.eval --questions benchmarks/questions/atlas_recall.yaml
    python -m benchmarks.eval --mode no_tools --questions ... --out results_no_tools.csv

Outputs:
    benchmarks/results/<timestamp>_<mode>.csv  -- per-question pass/fail + raw answer
    stdout                                     -- summary table with score
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from organoid_agent import (  # noqa: E402
    AgentLLM, LLMConfig,
    HNOCAModel, PubMedTool,
    Analysis, ResearchPlanning,
)


# --------------------------------------------------------------- #
# Grading                                                         #
# --------------------------------------------------------------- #
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?\s*%?")


def _extract_numbers(text: str) -> list[float]:
    """Pull all numbers from a string, converting percentages to decimals."""
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        s = m.group().strip()
        if s.endswith("%"):
            try:
                out.append(float(s[:-1].strip()) / 100.0)
            except ValueError:
                continue
        else:
            try:
                out.append(float(s))
            except ValueError:
                continue
    return out


def _within(observed: float, expected: float, tol: float | None) -> bool:
    tol = 0.05 if tol is None else tol
    return abs(observed - expected) <= max(tol, tol * abs(expected))


def _grade_scalar(answer_text: str, expected: float, tol: float | None) -> tuple[bool, str]:
    nums = _extract_numbers(answer_text)
    if not nums:
        return False, "no number found"
    best = min(nums, key=lambda x: abs(x - expected))
    return _within(best, expected, tol), f"best={best} vs expected={expected} (tol={tol})"


def _is_content_key(k: str) -> bool:
    """Content keys are cell-type names, gene names, protocol names —
    things that should be quoted literally in the answer. Metadata keys
    (n_labels, max_days, etc.) are snake_case identifiers we don't expect
    the model to repeat verbatim."""
    if not isinstance(k, str):
        return False
    if " " in k:
        return True
    if any(c.isupper() for c in k[1:]):
        return True
    if len(k) > 15:
        return True
    return False


def _grade_dict(answer_text: str, expected: dict, tol: float | None) -> tuple[bool, str]:
    """Per (key, value) check with stricter matching:
       - "min_fraction" / "min_<x>" / "max_<x>": HARD requirement (must pass)
       - content-style key: BOTH label-present AND value-close required
       - metadata-style key: value-close only
       - bool value: skipped (manual review)
    """
    detail = []
    nums = _extract_numbers(answer_text)
    required_passed = 0
    required_total = 0
    optional_passed = 0
    optional_total = 0

    for k, v in expected.items():
        is_required = isinstance(k, str) and (
            k.startswith("min_") or k.startswith("max_") or k == "label"
        )
        if isinstance(v, bool):
            detail.append(f"{k}: bool {v} — manual review")
            if is_required:
                required_passed += 1; required_total += 1
            else:
                optional_passed += 1; optional_total += 1
            continue
        if k == "min_fraction" and isinstance(v, (int, float)):
            satisfied = any(n >= float(v) for n in nums)
            detail.append(f"min_fraction>={v}: {satisfied}")
            required_total += 1; required_passed += int(satisfied)
            continue
        if isinstance(v, (int, float)):
            close = any(_within(n, float(v), tol) for n in nums) if nums else False
            if _is_content_key(k):
                label_in = k.lower() in answer_text.lower()
                this_ok = label_in and close
                detail.append(f"'{k}'~={v}: label={label_in} value={close}")
            else:
                this_ok = close
                detail.append(f"{k}~={v}: {close}")
            if is_required:
                required_total += 1; required_passed += int(this_ok)
            else:
                optional_total += 1; optional_passed += int(this_ok)
            continue
        if isinstance(v, str):
            present = v.lower() in answer_text.lower()
            detail.append(f"'{v}' present: {present}")
            if is_required:
                required_total += 1; required_passed += int(present)
            else:
                optional_total += 1; optional_passed += int(present)
            continue

    # Required must all pass; optional needs majority.
    req_ok = (required_total == 0) or (required_passed == required_total)
    opt_ok = (optional_total == 0) or (optional_passed >= (optional_total + 1) // 2)
    return req_ok and opt_ok, "; ".join(detail)


def grade(answer_text: str, expected: Any, tol: float | None) -> tuple[bool, str]:
    if isinstance(expected, (int, float)):
        return _grade_scalar(answer_text, float(expected), tol)
    if isinstance(expected, dict):
        return _grade_dict(answer_text, expected, tol)
    if isinstance(expected, list):
        return _grade_dict(answer_text, {str(i): v for i, v in enumerate(expected)}, tol)
    # categorical string
    return (str(expected).lower() in answer_text.lower(),
            f"looking for '{expected}' (substring match)")


# --------------------------------------------------------------- #
# Main                                                            #
# --------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="benchmarks/questions/atlas_recall.yaml")
    ap.add_argument("--mode", choices=["agent", "no_tools", "agent_plan"], default="agent",
                    help="agent = analysis-module with tools; "
                         "no_tools = analysis-module with tools disabled (LLM-only baseline); "
                         "agent_plan = research-planning + analysis (full pipeline minus literature)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="run only first N questions")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=True)
    except ImportError:
        pass

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); return 1

    llm = AgentLLM(LLMConfig(temperature=0.4), verbose=True)
    print(f"Loading questions from {args.questions} ...")
    with open(args.questions) as f:
        spec = yaml.safe_load(f)
    questions = spec["questions"]
    if args.limit:
        questions = questions[: args.limit]

    # Always load HNOCA — it's needed for the Analysis module's system prompt
    # (describe_inputs) even in no_tools mode.
    hnoca = HNOCAModel(verbose=True)
    pubmed = PubMedTool()
    analysis = Analysis(llm, hnoca, pubmed=pubmed, verbose=False)
    planning = ResearchPlanning(llm, verbose=False) if args.mode == "agent_plan" else None

    print(f"\nRunning {len(questions)} questions in mode={args.mode!r} ...\n")

    rows = []
    n_pass = 0
    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected = q["expected_answer"]
        tol = q.get("tolerance")
        expected_tool = q.get("expected_tool")

        print(f"--- {qid} ---")
        print(f"Q: {question}")

        plan_text = None
        if planning is not None:
            plan = planning(question)
            plan_text = plan.as_text()

        result = analysis.run(
            question,
            research_plan=plan_text,
            use_tools=(args.mode != "no_tools"),
        )
        answer = result.answer
        print(f"A: {answer[:300]}{'...' if len(answer) > 300 else ''}")
        if result.tool_calls:
            print(f"   tools used: {[t['name'] for t in result.tool_calls]}")
        if result.error:
            print(f"   ERROR: {result.error}")

        ok, detail = grade(answer, expected, tol)
        tool_called_correctly = (
            (expected_tool is None)
            or any(tc["name"] == expected_tool for tc in result.tool_calls)
        )
        n_pass += int(ok)
        print(f"   graded: {'PASS' if ok else 'FAIL'}  ({detail})")
        print(f"   tool match: {'yes' if tool_called_correctly else 'no'}")
        print()

        rows.append({
            "id": qid,
            "mode": args.mode,
            "question": question,
            "expected_answer": json.dumps(expected, default=str),
            "expected_tool": expected_tool or "",
            "tool_calls": "|".join(tc["name"] for tc in result.tool_calls),
            "tool_match": tool_called_correctly,
            "answer": answer,
            "pass": ok,
            "grade_detail": detail,
            "error": result.error or "",
        })

    # write csv
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "benchmarks" / "results" / f"{ts}_{args.mode}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    score = n_pass / len(questions)
    n_tool_match = sum(r["tool_match"] for r in rows)
    print("=" * 60)
    print(f"MODE: {args.mode}")
    print(f"PASS: {n_pass}/{len(questions)} = {score:.0%}")
    print(f"TOOL MATCH: {n_tool_match}/{len(questions)} = {n_tool_match/len(questions):.0%}")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
