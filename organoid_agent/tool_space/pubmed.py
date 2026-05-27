"""
PubMed search + abstract fetch via NCBI E-utilities.

Mirrors ``medea/tool_space/pubmed_search.py``. Two tools:
  - ``pubmed_search(query, retmax)`` : return list of {pmid, title, year}
  - ``fetch_abstract(pmid)``         : return abstract text for one PMID

NCBI's E-utilities are public — no key needed for low rates. If
``NCBI_API_KEY`` is set, requests are allowed at a higher rate.
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedTool:
    """Two-method PubMed tool with light caching."""

    def __init__(self, verbose: bool = False):
        self.api_key = os.environ.get("NCBI_API_KEY")
        self.verbose = verbose
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #
    def pubmed_search(self, query: str, retmax: int = 10) -> dict[str, Any]:
        """Return PubMed records matching the query.

        Returns:
            {"query": ..., "count": N, "results": [{pmid, title, year}, ...]}
        """
        cache_key = f"search:{query}:{retmax}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": int(retmax),
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            r = requests.get(ESEARCH_URL, params=params, timeout=15)
            r.raise_for_status()
            esearch = r.json()
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        pmids = esearch.get("esearchresult", {}).get("idlist", [])
        results = self._fetch_summaries(pmids) if pmids else []
        out = {
            "query": query,
            "count": int(esearch.get("esearchresult", {}).get("count", 0)),
            "results": results,
        }
        self._cache[cache_key] = out
        return out

    def _fetch_summaries(self, pmids: list[str]) -> list[dict[str, Any]]:
        """Pull title + year for a list of PMIDs via efetch."""
        if not pmids:
            return []
        params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            time.sleep(0.34)  # NCBI rate-limit safe (3 req/s)
            r = requests.get(EFETCH_URL, params=params, timeout=20)
            r.raise_for_status()
            root = ET.fromstring(r.text)
        except Exception:
            return [{"pmid": p, "title": "(fetch failed)", "year": ""} for p in pmids]

        out: list[dict[str, Any]] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = (art.findtext(".//PMID") or "").strip()
            title = (art.findtext(".//ArticleTitle") or "").strip()
            year = (art.findtext(".//PubDate/Year")
                    or art.findtext(".//PubDate/MedlineDate")
                    or "").strip()[:4]
            journal = (art.findtext(".//Journal/Title")
                       or art.findtext(".//Journal/ISOAbbreviation")
                       or "").strip()
            out.append({"pmid": pmid, "title": title, "year": year, "journal": journal})
        return out

    # ------------------------------------------------------------------ #
    # Abstract fetch                                                     #
    # ------------------------------------------------------------------ #
    def fetch_abstract(self, pmid: str) -> dict[str, Any]:
        """Return the abstract text for one PMID."""
        cache_key = f"abs:{pmid}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        params: dict[str, Any] = {
            "db": "pubmed", "id": str(pmid),
            "retmode": "xml", "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            time.sleep(0.34)
            r = requests.get(EFETCH_URL, params=params, timeout=20)
            r.raise_for_status()
            root = ET.fromstring(r.text)
        except Exception as exc:
            return {"pmid": pmid, "error": f"{type(exc).__name__}: {exc}"}

        art = root.find(".//PubmedArticle")
        if art is None:
            return {"pmid": pmid, "error": "PMID not found"}
        title = (art.findtext(".//ArticleTitle") or "").strip()
        # Abstract can have multiple labelled sections
        sections = []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.get("Label", "")
            text = ("".join(ab.itertext()) or "").strip()
            if text:
                sections.append(f"{label + ': ' if label else ''}{text}")
        abstract = "\n\n".join(sections)
        journal = (art.findtext(".//Journal/Title")
                   or art.findtext(".//Journal/ISOAbbreviation")
                   or "").strip()
        year = (art.findtext(".//PubDate/Year")
                or art.findtext(".//PubDate/MedlineDate") or "").strip()[:4]
        out = {
            "pmid": str(pmid), "title": title, "year": year,
            "journal": journal, "abstract": abstract,
        }
        self._cache[cache_key] = out
        return out
