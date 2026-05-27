"""
Literature-Reasoning module — third stage of the Medea-style agent.

Wraps the PubMed tool with an LLM that filters and synthesises retrieved
abstracts. Mirrors ``medea.modules.literature_reasoning``.

Used by the agent for citation-grounding: "name three papers that support
claim X" type questions. Returns a list of (pmid, title, evidence) tuples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..tool_space.pubmed import PubMedTool
from .agent_llms import AgentLLM


@dataclass
class CitedClaim:
    claim: str
    pmid: str
    title: str
    evidence: str  # one-sentence quote / paraphrase justifying the citation


class LiteratureReasoning:
    """Search PubMed for papers supporting (or refuting) a claim."""

    def __init__(self, llm: AgentLLM, pubmed: PubMedTool, verbose: bool = True):
        self.llm = llm
        self.pubmed = pubmed
        self.verbose = verbose

    def support_claim(self, claim: str, k: int = 5) -> list[CitedClaim]:
        """For a given claim, find up to k supporting PubMed papers and
        extract a single-sentence justification from each abstract."""
        if self.verbose:
            print(f"[LiteratureReasoning] searching PubMed for: {claim}")
        hits = self.pubmed.pubmed_search(query=claim, retmax=k)
        cited: list[CitedClaim] = []
        for h in hits.get("results", []):
            pmid = h.get("pmid", "")
            title = h.get("title", "")
            abstract = self.pubmed.fetch_abstract(pmid=pmid).get("abstract", "")
            evidence = self._extract_evidence(claim, title, abstract)
            cited.append(CitedClaim(claim=claim, pmid=pmid, title=title, evidence=evidence))
        return cited

    def _extract_evidence(self, claim: str, title: str, abstract: str) -> str:
        """LLM call: pull the single best supporting sentence from the abstract."""
        if not abstract:
            return ""
        messages = [
            {"role": "system", "content":
             "You are an expert at finding evidence in scientific abstracts. "
             "Given a CLAIM and an ABSTRACT, return the ONE sentence from the "
             "abstract that most directly supports the claim, verbatim. If no "
             "sentence supports the claim, return 'NO SUPPORT'. Output the "
             "sentence and nothing else."},
            {"role": "user", "content": f"CLAIM: {claim}\n\nTITLE: {title}\n\nABSTRACT: {abstract}"},
        ]
        try:
            resp = self.llm.chat(messages)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"(extraction error: {type(exc).__name__})"
