#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated ArXiv Reviewer → Deep JSON → Arweave (Bundlr) → GraphQL

This module implements a one‑file pipeline that can fetch papers from
arXiv, extract their contents, generate structured reviews using
OpenAI models, optionally upload the results to Arweave via Bundlr,
produce benchmark charts, and emit a GraphQL schema for local
consumption.  The emphasis is on producing a deeply nested JSON
representation of each paper with rich provenance metadata.

Features:
  • Fetch the latest arXiv preprints for a given category (default
    ``cs.LO``) or process specific arXiv IDs.
  • Download PDFs and extract text page‑by‑page via PyMuPDF
    (``fitz``), with a fallback to PyPDF2 if unavailable.
  • Use a multi‑model strategy with OpenAI to plan, review, verify
    and format the assessment into a strict JSON schema.
  • Persist each assessment as a JSON file and, optionally, upload
    it to Arweave via Bundlr with appropriate metadata tags.
  • Optionally generate benchmark charts summarising readiness
    levels and average scores across a corpus of assessments.
  • Optionally produce a GraphQL schema and run a local GraphQL
    endpoint to explore the generated data.

Usage examples::

    python arxiv_superreview.py --category cs.LO --num-papers 100 \
        --openai-key sk-... --output-dir out --benchmark --generate-graphql

    python arxiv_superreview.py --ids 2501.01234 2501.04321 \
        --openai-key sk-... --output-dir out --upload \
        --bundlr-wallet /path/wallet.json --bundlr-currency arweave

Requirements:
    pip install requests PyMuPDF PyPDF2 openai seaborn matplotlib graphene flask-graphql tqdm

Notes:
  • Only the reviewer model (e.g. ``gpt-5``) performs any evaluation;
    the planner/verifier/formatter models are non‑evaluative.
  • When generating charts, the script strives to avoid clutter by
    using consistent fonts, legible labels and high DPI settings.
  • The JSON repair pass with the formatter model is non‑evaluative
    and exists solely to enforce schema compliance.

Disclaimer:
    This script calls external services (arXiv, OpenAI, Bundlr).  Make
    sure you have appropriate API keys and respect applicable rate
    limits and terms of use.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests
from tqdm import tqdm

# Optional dependencies
try:
    import fitz  # type: ignore  # PyMuPDF
    HAVE_FITZ = True
except Exception:
    HAVE_FITZ = False

try:
    import PyPDF2  # type: ignore
    HAVE_PYPDF2 = True
except Exception:
    HAVE_PYPDF2 = False

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None  # type: ignore

try:
    import matplotlib
    import matplotlib.pyplot as plt  # type: ignore
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

try:
    from openai import OpenAI
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

# Configuration Defaults
ARXIV_API: str = "https://export.arxiv.org/api/query"
UA: str = "arxiv-superreview/1.0 (+https://example.org)"
DEFAULT_CATEGORY: str = "cs.LO"

# Model roles: overridable by environment variables
MODELS: Dict[str, str] = {
    "planner": os.getenv("PLANNER_MODEL", "o3"),
    "reviewer": os.getenv("REVIEW_MODEL", "gpt-5"),
    "verifier": os.getenv("VERIFIER_MODEL", "o1"),
    "formatter": os.getenv("FORMATTER_MODEL", "gpt-5-nano"),
}

# Aesthetic defaults for charts
PLOT_DPI: int = 180
PLOT_STYLE: str = "whitegrid"
PLOT_FONT_SIZE: int = 10

# Data classes

@dataclass
class PaperMetadata:
    """
    Metadata describing a paper fetched from arXiv.

    The ``license`` field is intended to capture the usage license for the
    paper on arXiv.  By default this value is an empty string and
    populated later in the pipeline via ``get_arxiv_license``.
    """
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str
    published: str
    authors: List[str]
    categories: List[str]
    license: str = ""


@dataclass
class PageInfo:
    """Per‑page information recorded during PDF extraction."""
    page_no: int
    char_len: int
    sha256: str


@dataclass
class Location:
    """Location information used in claim rows and other annotations."""
    pages: List[int]
    section: Optional[str] = None
    figure: Optional[str] = None
    table: Optional[str] = None


@dataclass
class ClaimRow:
    """Entry for a single claim in the claims matrix."""
    claim: str
    location: Location
    evidence: str
    strength: str
    missing: str


@dataclass
class ScoreItem:
    """Scorecard entry for a single evaluation dimension."""
    dimension: str
    score: float
    rationale: str


@dataclass
class Readiness:
    """Readiness level of a paper."""
    status: str
    justification: str


@dataclass
class AssessmentDeep:
    """Type hints for a deep assessment JSON structure."""
    executive_summary: Dict[str, Any]
    claims_matrix: Dict[str, Any]
    methods_evidence_audit: Dict[str, Any]
    novelty_significance: Dict[str, Any]
    clarity_organization: Dict[str, Any]
    limitations_risks: List[Dict[str, Any]]
    actionable_improvements: List[Dict[str, Any]]
    quality_scorecard: List[ScoreItem]
    readiness_level: Readiness
    questions_to_authors: List[Dict[str, Any]]
    line_page_anchored_notes: List[Dict[str, Any]]


# Helper utilities

def safe_mkdir(p: Path) -> None:
    """Create a directory and parents if necessary."""
    p.mkdir(parents=True, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    """Compute the SHA‑256 of a bytes sequence."""
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    """Compute the SHA‑256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_code_fences(s: str) -> str:
    """Remove Markdown code fences from a string (if present)."""
    s = s.strip()
    s = re.sub(r"^`(?:json)?\s*", "", s)
    s = re.sub(r"\s*`$", "", s)
    return s.strip()


def json_loads_strict(s: str) -> Any:
    """Load JSON from a string, stripping Markdown fences if necessary."""
    return json.loads(strip_code_fences(s))


def now_iso() -> str:
    """Return current UTC time in ISO‑8601 format using a timezone-aware object."""
    # Use timezone-aware datetime to avoid deprecation warnings
    return dt.datetime.now(dt.timezone.utc).isoformat()


def shorten(s: str, n: int = 80) -> str:
    """Shorten a string to ``n`` characters, replacing newlines with spaces."""
    s = s.replace("\n", " ")
    return (s[: n - 1] + "…") if len(s) > n else s


def sleep_backoff(tries: int) -> None:
    """Sleep with exponential backoff up to 10 seconds."""
    time.sleep(min(10, 1.5 ** tries))


# arXiv integration

def fetch_arxiv_feed(category: str, start: int, max_results: int) -> str:
    """Fetch an Atom feed from arXiv for a given category."""
    params = {
        "search_query": f"cat:{category}",
        "start": str(start),
        "max_results": str(max_results),
    }
    r = requests.get(ARXIV_API, params=params, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    return r.text


def parse_arxiv_feed(atom_xml: str) -> List[PaperMetadata]:
    """Parse an arXiv Atom feed into a list of PaperMetadata."""
    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(atom_xml)
    out: List[PaperMetadata] = []
    for e in root.findall("atom:entry", ns):
        id_el = e.find("atom:id", ns)
        title_el = e.find("atom:title", ns)
        summary_el = e.find("atom:summary", ns)
        pub_el = e.find("atom:published", ns)
        authors: List[str] = [
            a.findtext("atom:name", default="", namespaces=ns).strip()
            for a in e.findall("atom:author", ns)
        ]
        cats: List[str] = [c.attrib.get("term", "") for c in e.findall("atom:category", ns)]
        pdf_url = ""
        for link in e.findall("atom:link", ns):
            t = link.attrib.get("type", "")
            title = link.attrib.get("title", "")
            if t == "application/pdf" or (title and title.lower() == "pdf"):
                pdf_url = link.attrib.get("href", "")
                break
        arxiv_id = (id_el.text or "").rsplit("/", 1)[-1] if id_el is not None else ""
        out.append(
            PaperMetadata(
                arxiv_id=arxiv_id,
                title=(title_el.text or "").strip(),
                abstract=(summary_el.text or "").strip(),
                pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published=(pub_el.text or "").strip(),
                authors=[a for a in authors if a],
                categories=[c for c in cats if c],
            )
        )
    # Local sort by published date (descending)
    def parse_pub(p: str) -> dt.datetime:
        with contextlib.suppress(Exception):
            return dt.datetime.strptime(p, "%Y-%m-%dT%H:%M:%SZ")
        return dt.datetime.min
    out.sort(key=lambda m: parse_pub(m.published), reverse=True)
    return out


def download_pdf(url: str, dest: Path) -> Path:
    """Download a PDF from arXiv if it does not already exist."""
    safe_mkdir(dest.parent)
    if dest.exists():
        return dest
    with requests.get(url, stream=True, headers={"User-Agent": UA}, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return dest


def get_arxiv_license(arxiv_id: str) -> str:
    """
    Fetch the license associated with an arXiv paper.

    This implementation is a placeholder: it does not query arXiv and
    always returns the string "Unknown".  arXiv does not expose
    licence information via the Atom feed.  A future enhancement could
    retrieve the licence from the paper's HTML page or another API.
    """
    # Placeholder: always return Unknown since we cannot fetch licences
    return "Unknown"


def extract_text_per_page(pdf_path: Path) -> Tuple[List[str], List[PageInfo]]:
    """Extract text per page from a PDF using PyMuPDF or PyPDF2."""
    if HAVE_FITZ:
        doc = fitz.open(pdf_path)  # type: ignore[arg-type]
        pages_text: List[str] = []
        pages_meta: List[PageInfo] = []
        for i, page in enumerate(doc, start=1):
            t = page.get_text()
            pages_text.append(t)
            pages_meta.append(PageInfo(page_no=i, char_len=len(t), sha256=sha256_bytes(t.encode("utf-8"))))
        doc.close()
        return pages_text, pages_meta
    elif HAVE_PYPDF2:
        reader = PyPDF2.PdfReader(str(pdf_path))  # type: ignore[arg-type]
        pages_text: List[str] = []
        pages_meta: List[PageInfo] = []
        for i, page in enumerate(reader.pages, start=1):
            with contextlib.suppress(Exception):
                t = page.extract_text() or ""
                if not t:
                    t = ""
                pages_text.append(t)
                pages_meta.append(PageInfo(page_no=i, char_len=len(t), sha256=sha256_bytes(t.encode("utf-8"))))
        return pages_text, pages_meta
    else:
        raise RuntimeError("Neither PyMuPDF nor PyPDF2 is installed; cannot extract PDF text.")


# OpenAI multi‑model client

class MultiModel:
    """Helper for orchestrating multiple OpenAI models for planning, reviewing, verifying and formatting."""

    def __init__(self, api_key: str, planner: str, reviewer: str, verifier: str, formatter: str) -> None:
        if not HAVE_OPENAI:
            raise RuntimeError("openai package not installed. pip install openai")
        self.client = OpenAI(api_key=api_key)
        self.models = {
            "planner": planner,
            "reviewer": reviewer,
            "verifier": verifier,
            "formatter": formatter,
        }

        # Track whether the OpenAI backend supports certain optional parameters.  Some
        # models reject parameters like ``max_tokens`` or a ``temperature`` value of
        # zero.  When a call fails due to an unsupported parameter, the corresponding
        # flag will be set to ``False`` so subsequent calls omit that parameter.
        # A warning is printed only once per parameter per instance.
        self.max_tokens_supported: bool = True
        self._max_tokens_warning_printed: bool = False
        self.temperature_supported: bool = True
        self._temperature_warning_printed: bool = False

    def _chat_create(self,
                     model_key: str,
                     messages: List[Dict[str, Any]],
                     temperature: float,
                     max_tokens: Optional[int] = None,
                     **extra: Any) -> Any:
        """
        Helper to call the OpenAI chat API with optional ``max_tokens`` handling.

        If the backend does not support the ``max_tokens`` parameter, this method
        will detect the error, disable passing the parameter, print a one‑time
        warning and retry without ``max_tokens``.  Additional keyword arguments
        (such as ``reasoning_effort``) are forwarded unchanged.
        """
        # Build base parameters
        params: Dict[str, Any] = {
            "model": self.models[model_key],
            "messages": messages,
        }
        # Include temperature only if supported. Some models reject temperature values
        # other than the default (1). The ``temperature`` argument is passed in
        # explicitly to this helper, so check our tracking flag before adding it.
        if self.temperature_supported:
            params["temperature"] = temperature
        # Include max_tokens only if supported and provided
        if self.max_tokens_supported and max_tokens is not None:
            params["max_tokens"] = max_tokens
        # Merge in any extra parameters
        params.update(extra)
        try:
            return self.client.chat.completions.create(**params)
        except Exception as e:
            msg = str(e)
            # Detect unsupported max_tokens
            if self.max_tokens_supported and "max_tokens" in msg and "unsupported" in msg:
                # Disable for future calls
                self.max_tokens_supported = False
                if not self._max_tokens_warning_printed:
                    print(
                        '[warn] The "max_tokens" parameter is not supported by the selected model; disabling it for future calls'
                    )
                    self._max_tokens_warning_printed = True
                # Remove max_tokens and retry
                params.pop("max_tokens", None)
                return self.client.chat.completions.create(**params)
            # Detect unsupported temperature (e.g. temperature=0 not allowed)
            if self.temperature_supported and "temperature" in msg and ("unsupported" in msg or "does not support" in msg):
                # Disable for future calls and remove the temperature parameter
                self.temperature_supported = False
                if not self._temperature_warning_printed:
                    print(
                        '[warn] The specified "temperature" is not supported by the selected model; using the default value for future calls'
                    )
                    self._temperature_warning_printed = True
                params.pop("temperature", None)
                return self.client.chat.completions.create(**params)
            # Otherwise propagate error
            raise

    # Planner (o3)
    def plan_checklist(self, title: str, abstract: str, pages_digest: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Use the planner model to produce a review plan and evidence checklist."""
        msg_user = (
            "You are preparing a plan for reviewing a research paper.\n"
            "Return STRICT JSON (no prose) with keys: {\"object\":\"plan\","
            "\"sections\":[{\"name\":\"...\",\"what_to_look_for\":[...] }],"
            "\"checklist\":[\"...\"],\"likely_relevant_pages\":[{"
            "\"page\":int,\"priority\":\"high|medium|low\"}],\"notes\":\"short\"}.\n"
            "Use ONLY the abstract/title and page lengths. No evaluation.\n\n"
            f"TITLE: {title}\n\nABSTRACT:\n{abstract}\n\n"
            "PAGES DIGEST (page_no, char_count):\n"
            + json.dumps([{"page": p, "len": l} for (p, l) in pages_digest], ensure_ascii=False)
        )
        r = self._chat_create(
            model_key="planner",
            messages=[
                {"role": "system", "content": "You are a meticulous planning assistant. Respond in strict JSON only."},
                {"role": "user", "content": msg_user},
            ],
            temperature=0.0,
            max_tokens=900,
        )
        txt = r.choices[0].message.content or "{}"
        with contextlib.suppress(Exception):
            return json_loads_strict(txt)
        return {"object": "plan", "sections": [], "checklist": [], "likely_relevant_pages": [], "notes": ""}

    # Reviewer (gpt‑5)
    def review_final_json(
        self,
        paper: PaperMetadata,
        per_page_texts: List[str],
        checklist_plan: Dict[str, Any],
        max_tokens: int = 3500,
    ) -> Dict[str, Any]:
        """Ask the reviewer model to read the full paper text and produce a deep JSON assessment."""
        schema_hint: Dict[str, Any] = {
            "metadata": {
                "arxiv_id": "str",
                "title": "str",
                "authors": ["str"],
                "published": "ISO-8601",
                "categories": ["str"],
                "pdf_sha256": "str",
            },
            "provenance": {
                "pipeline": {
                    "planner_model": "str",
                    "review_model": "str",
                    "verifier_model": "str",
                    "formatter_model": "str",
                },
                "timestamps": {
                    "planned_at": "ISO-8601",
                    "reviewed_at": "ISO-8601",
                },
                "parameters": {
                    "temperature": 0,
                    "prompt_version": "v1",
                    "notes": "str optional",
                },
            },
            "document_structure": {
                "pages": [
                    {"page_no": "int", "char_len": "int", "sha256": "str"},
                ],
                "sections": [
                    {"title": "str", "page_start": "int", "page_end": "int", "children": []},
                ],
            },
            "review": {
                "executive_summary": {"text": "markdown"},
                "claims_matrix": {
                    "columns": ["Claim", "Where", "Evidence", "Strength", "Missing"],
                    "rows": [
                        {
                            "claim": "str",
                            "location": {
                                "pages": [1],
                                "section": "str?",
                                "figure": "str?",
                                "table": "str?",
                            },
                            "evidence": "str",
                            "strength": "Low|Medium|High",
                            "missing": "str",
                        }
                    ],
                },
                "methods_evidence_audit": {
                    "design_setup": "markdown",
                    "statistical_validity": "markdown",
                    "threats_to_validity": {
                        "internal": "markdown",
                        "external": "markdown",
                        "construct": "markdown",
                    },
                    "reproducibility": "markdown",
                    "if_theoretical": {
                        "definitions": "markdown",
                        "lemmas": "markdown",
                        "proof_gaps": "markdown",
                        "boundary_cases": "markdown",
                        "counterexamples": "markdown",
                    },
                    "if_computational_ml": {
                        "splits": "markdown",
                        "ablations": "markdown",
                        "calibration": "markdown",
                        "robustness": "markdown",
                        "fairness_harms": "markdown",
                    },
                    "if_qualitative_mixed": {
                        "sampling": "markdown",
                        "coding_scheme": "markdown",
                        "saturation": "markdown",
                        "triangulation": "markdown",
                        "reflexivity": "markdown",
                    },
                    "if_biomedical_human": {
                        "approvals_consent": "markdown",
                        "privacy": "markdown",
                        "risk_mgmt": "markdown",
                        "preregistration": "markdown",
                    },
                },
                "novelty_significance": {
                    "what_is_new": "markdown",
                    "missing_comparisons": ["markdown"],
                },
                "clarity_organization": {
                    "title_abstract_alignment": "markdown",
                    "structure_flow": "markdown",
                    "figures_tables_readability": "markdown",
                    "terminology_consistency": "markdown",
                    "ambiguities": "markdown",
                },
                "limitations_risks": [
                    {
                        "item": "markdown",
                        "type": "limitation|risk",
                        "location": {"pages": [1]},
                    }
                ],
                "actionable_improvements": [
                    {
                        "priority": 1,
                        "change": "str",
                        "where": "str",
                        "rationale": "str",
                        "expected_impact": "str",
                    }
                ],
                "quality_scorecard": [
                    {
                        "dimension": "Soundness|Strength of evidence|Originality|Importance of contribution|Clarity of writing|Reproducibility|Presentation|Resource transparency|Safety/ethics considerations|Overall",
                        "score": 0.0,
                        "rationale": "short",
                    }
                ],
                "readiness_level": {
                    "status": "Ready|Minor adjustments needed|Major adjustments needed|Not ready",
                    "justification": "markdown",
                },
                "questions_to_authors": [
                    {"question": "str", "location": {"pages": [1]}},
                ],
                "line_page_anchored_notes": [
                    {"page": 1, "anchor": "str", "note": "str"},
                ],
            },
            "verification": {
                "o1_consistency_checks": [],
                "json_validated": False,
            },
            "arweave": {
                "tags": [
                    {"name": "arxiv_id", "value": paper.arxiv_id},
                    {"name": "Content-Type", "value": "application/json"},
                    {"name": "category", "value": (paper.categories[0] if paper.categories else "")},
                    {"name": "app", "value": "arxiv-superreview"},
                ]
            },
        }
        # Prepare input content: keep page boundaries; prepend a minimal plan.
        page_bundle = [
            {"page_no": i + 1, "text": per_page_texts[i]}
            for i in range(len(per_page_texts))
        ]
        instruction = (
            "You are an expert reviewer. Read the full paper text provided page-by-page.\n"
            "Output STRICT JSON only (no markdown fences, no prose). Your JSON MUST conform to the provided schema template.\n"
            "Every factual statement must be grounded in the text and, whenever possible, include page references.\n"
            "Follow neutral, precise language. If a section is N/A, include an empty string or empty list.\n"
            "Do NOT invent content that is not in the text. Do NOT summarise outside the document.\n"
            "IMPORTANT: ensure numeric scores are 0–10 floats; include short rationales; include readiness level.\n"
            "IMPORTANT: the `claims_matrix.rows[*].location.pages` MUST use 1-based page numbers.\n"
        )
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "SCHEMA TEMPLATE:\n" + json.dumps(schema_hint, ensure_ascii=False)},
            {"role": "user", "content": "PLANNING CHECKLIST (from planner, non-binding):\n" + json.dumps(checklist_plan, ensure_ascii=False)},
            {"role": "user", "content": "PAPER PAGES (use these only):\n" + json.dumps(page_bundle, ensure_ascii=False)},
            {"role": "user", "content": "PAPER METADATA:\n" + json.dumps(dataclasses.asdict(paper), ensure_ascii=False)},
        ]
        tries = 0
        while True:
            try:
                r = self._chat_create(
                    model_key="reviewer",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                raw = r.choices[0].message.content or "{}"
                return json_loads_strict(raw)
            except Exception:
                tries += 1
                if tries >= 3:
                    raise
                sleep_backoff(tries)

    # Verifier (o1)
    def verify_consistency(self, assessment_json: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the verifier model to point out internal consistency issues."""
        instr = (
            "You are a verifier. Inspect the provided JSON assessment for internal consistency ONLY.\n"
            "Return STRICT JSON with keys: {\"object\":\"verification\",\"issues\":[{\"field\":\"path\",\"problem\":\"...\",\"severity\":\"low|medium|high\"}],\"notes\":\"short\"}.\n"
            "Do NOT re-evaluate the paper. Do NOT modify the assessment."
        )
        tries = 0
        while True:
            try:
                r = self._chat_create(
                    model_key="verifier",
                    messages=[
                        {"role": "system", "content": instr},
                        {"role": "user", "content": json.dumps(assessment_json, ensure_ascii=False)},
                    ],
                    temperature=0.0,
                    max_tokens=800,
                    reasoning_effort="medium",
                )
                return json_loads_strict(r.choices[0].message.content or "{}")
            except Exception:
                tries += 1
                if tries >= 2:
                    return {"object": "verification", "issues": [], "notes": "verification skipped after retries"}
                sleep_backoff(tries)

    # Formatter (gpt‑5‑nano)
    def repair_json_to_schema(self, assessment_json: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Use the formatter model to repair malformed assessment JSON to the schema."""
        if isinstance(assessment_json, dict):
            assessment_json = json.dumps(assessment_json, ensure_ascii=False)
        instruction = (
            "You are a JSON formatter. Convert the input JSON into a VALID JSON object that conforms to the "
            "schema keys and types. Fill missing keys with empty strings/lists/objects as appropriate. "
            "Do NOT invent new facts; only restructure/rename to fit the schema. Output STRICT JSON only."
        )
        tries = 0
        while True:
            try:
                r = self._chat_create(
                    model_key="formatter",
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": assessment_json},
                    ],
                    temperature=0.0,
                    max_tokens=1500,
                )
                return json_loads_strict(r.choices[0].message.content or "{}")
            except Exception:
                tries += 1
                if tries >= 2:
                    # Final fallback: best-effort local parse
                    with contextlib.suppress(Exception):
                        return json.loads(strip_code_fences(assessment_json))
                    return {"error": "Could not repair JSON"}


# JSON validation / shaping

def ensure_deep_schema_shape(
    obj: Dict[str, Any],
    paper: PaperMetadata,
    pdf_sha256: str,
    pages_meta: List[PageInfo],
    planner_model: str,
    reviewer_model: str,
    verifier_model: str,
    formatter_model: str,
    planned_at: str,
    reviewed_at: str,
) -> Dict[str, Any]:
    """Ensure required top‑level keys exist and have proper shapes; fill missing defaults non‑destructively."""
    def ensure(d: Dict[str, Any], key: str, default: Any) -> None:
        if key not in d or d[key] is None:
            d[key] = default

    ensure(obj, "metadata", {})
    md = obj["metadata"]
    md.setdefault("arxiv_id", paper.arxiv_id)
    md.setdefault("title", paper.title)
    md.setdefault("authors", paper.authors)
    md.setdefault("published", paper.published)
    md.setdefault("categories", paper.categories)
    md.setdefault("pdf_sha256", pdf_sha256)
    md.setdefault("license", paper.license)

    ensure(obj, "provenance", {})
    pv = obj["provenance"]
    pv.setdefault("pipeline", {})
    pv["pipeline"].setdefault("planner_model", planner_model)
    pv["pipeline"].setdefault("review_model", reviewer_model)
    pv["pipeline"].setdefault("verifier_model", verifier_model)
    pv["pipeline"].setdefault("formatter_model", formatter_model)
    pv.setdefault("timestamps", {})
    pv["timestamps"].setdefault("planned_at", planned_at)
    pv["timestamps"].setdefault("reviewed_at", reviewed_at)
    pv.setdefault("parameters", {"temperature": 0, "prompt_version": "v1"})

    ensure(obj, "document_structure", {})
    ds = obj["document_structure"]
    ds.setdefault(
        "pages",
        [
            {
                "page_no": p.page_no,
                "char_len": p.char_len,
                "sha256": p.sha256,
            }
            for p in pages_meta
        ],
    )
    ds.setdefault("sections", [])

    ensure(obj, "review", {})
    rv = obj["review"]
    rv.setdefault("executive_summary", {"text": ""})
    rv.setdefault(
        "claims_matrix",
        {
            "columns": ["Claim", "Where", "Evidence", "Strength", "Missing"],
            "rows": [],
        },
    )
    rv.setdefault(
        "methods_evidence_audit",
        {
            "design_setup": "",
            "statistical_validity": "",
            "threats_to_validity": {"internal": "", "external": "", "construct": ""},
            "reproducibility": "",
            "if_theoretical": {"definitions": "", "lemmas": "", "proof_gaps": "", "boundary_cases": "", "counterexamples": ""},
            "if_computational_ml": {"splits": "", "ablations": "", "calibration": "", "robustness": "", "fairness_harms": ""},
            "if_qualitative_mixed": {"sampling": "", "coding_scheme": "", "saturation": "", "triangulation": "", "reflexivity": ""},
            "if_biomedical_human": {"approvals_consent": "", "privacy": "", "risk_mgmt": "", "preregistration": ""},
        },
    )
    rv.setdefault("novelty_significance", {"what_is_new": "", "missing_comparisons": []})
    rv.setdefault(
        "clarity_organization",
        {
            "title_abstract_alignment": "",
            "structure_flow": "",
            "figures_tables_readability": "",
            "terminology_consistency": "",
            "ambiguities": "",
        },
    )
    rv.setdefault("limitations_risks", [])
    rv.setdefault("actionable_improvements", [])
    rv.setdefault("quality_scorecard", [])
    rv.setdefault("readiness_level", {"status": "", "justification": ""})
    rv.setdefault("questions_to_authors", [])
    rv.setdefault("line_page_anchored_notes", [])

    ensure(obj, "verification", {"o1_consistency_checks": [], "json_validated": False})
    ensure(
        obj,
        "arweave",
        {
            "tags": [
                {"name": "arxiv_id", "value": paper.arxiv_id},
                {"name": "Content-Type", "value": "application/json"},
                {"name": "category", "value": (paper.categories[0] if paper.categories else "")},
                {"name": "app", "value": "arxiv-superreview"},
            ]
        },
    )
    return obj


# Benchmarking charts

def generate_benchmark_chart(json_dir: Path, out_path: Path) -> None:
    """Generate a benchmark chart summarising readiness distribution and average scores."""
    if not HAVE_MPL:
        print("matplotlib not installed; skipping benchmark chart.")
        return
    files = sorted(json_dir.glob("*.json"))
    readiness_counts: Dict[str, int] = {
        "Ready": 0,
        "Minor adjustments needed": 0,
        "Major adjustments needed": 0,
        "Not ready": 0,
        "Unspecified": 0,
    }
    score_sums: Dict[str, float] = {}
    score_counts: Dict[str, int] = {}
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        status = obj.get("review", {}).get("readiness_level", {}).get("status", "") or "Unspecified"
        if status not in readiness_counts:
            status = "Unspecified"
        readiness_counts[status] += 1
        for si in obj.get("review", {}).get("quality_scorecard", []):
            dim = si.get("dimension", "Unknown")
            try:
                sc = float(si.get("score", 0.0))
            except Exception:
                sc = 0.0
            score_sums[dim] = score_sums.get(dim, 0.0) + sc
            score_counts[dim] = score_counts.get(dim, 0) + 1
    avg_scores = {k: (score_sums[k] / score_counts[k]) for k in score_sums.keys()}
    # Sort dimensions by value
    avg_sorted = sorted(avg_scores.items(), key=lambda kv: kv[1])
    if sns:
        sns.set_theme(style=PLOT_STYLE)  # type: ignore[attr-defined]
    plt.rcParams.update({"font.size": PLOT_FONT_SIZE})  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # type: ignore[attr-defined]
    # Left: readiness distribution
    labels = list(readiness_counts.keys())
    counts = [readiness_counts[k] for k in labels]
    axes[0].bar(labels, counts)
    axes[0].set_title("Readiness Level Distribution")
    axes[0].set_xlabel("Readiness Level")
    axes[0].set_ylabel("Number of Papers")
    for i, c in enumerate(counts):
        axes[0].text(i, c + max(1, 0.02 * (max(counts) if counts else 1)), str(c), ha="center", va="bottom", fontsize=9)
    axes[0].tick_params(axis="x", rotation=15)
    # Right: average scores
    dims = [k for k, _ in avg_sorted]
    vals = [v for _, v in avg_sorted]
    axes[1].barh(dims, vals)
    axes[1].set_title("Average Quality Scores by Dimension")
    axes[1].set_xlabel("Average Score (0–10)")
    for y, v in enumerate(vals):
        axes[1].text(v + 0.1, y, f"{v:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[attr-defined]


# GraphQL generation

# The following constants are used to write out a ``schema.py`` file which
# exposes the data via GraphQL. They are defined as plain strings to
# avoid syntax errors due to nested triple quotes. The header includes
# a small docstring and the imports needed to load the JSON files into
# the ``DATA`` dict.
GRAPHQL_HEADER = (
    '"""\n'
    'Auto-generated GraphQL schema for arXiv SuperReview.\n'
    'Expose deep hierarchical fields for each paper assessment.\n'
    '"""\n'
    'import json\n'
    'import graphene\n'
    'from pathlib import Path\n\n'
    'DATA_DIR = Path(__file__).parent / "json"\n\n'
    'def _load_all():\n'
    '    data = {}\n'
    '    if DATA_DIR.exists():\n'
    '        for fp in DATA_DIR.glob("*.json"):\n'
    '            with open(fp, "r", encoding="utf-8") as f:\n'
    '                try:\n'
    '                    obj = json.load(f)\n'
    '                except Exception:\n'
    '                    continue\n'
    '                key = obj.get("metadata", {}).get("arxiv_id", fp.stem)\n'
    '                data[key] = obj\n'
    '    return data\n\n'
    'DATA = _load_all()\n'
)

# The GraphQL schema definition is stored in ``GRAPHQL_SCHEMA``. It
# defines GraphQL object types corresponding to the nested JSON
# structure and exposes two query fields: ``paper(arxiv_id)`` and
# ``all_ids``.
GRAPHQL_SCHEMA = (
    'class Tag(graphene.ObjectType):\n'
    '    name = graphene.String()\n'
    '    value = graphene.String()\n\n'
    'class Location(graphene.ObjectType):\n'
    '    pages = graphene.List(graphene.Int)\n'
    '    section = graphene.String()\n'
    '    figure = graphene.String()\n'
    '    table = graphene.String()\n\n'
    'class ClaimRow(graphene.ObjectType):\n'
    '    claim = graphene.String()\n'
    '    location = graphene.Field(Location)\n'
    '    evidence = graphene.String()\n'
    '    strength = graphene.String()\n'
    '    missing = graphene.String()\n\n'
    'class ClaimsMatrix(graphene.ObjectType):\n'
    '    columns = graphene.List(graphene.String)\n'
    '    rows = graphene.List(ClaimRow)\n\n'
    'class ThreatsToValidity(graphene.ObjectType):\n'
    '    internal = graphene.String()\n'
    '    external = graphene.String()\n'
    '    construct = graphene.String()\n\n'
    'class MethodsTheoretical(graphene.ObjectType):\n'
    '    definitions = graphene.String()\n'
    '    lemmas = graphene.String()\n'
    '    proof_gaps = graphene.String()\n'
    '    boundary_cases = graphene.String()\n'
    '    counterexamples = graphene.String()\n\n'
    'class MethodsComputational(graphene.ObjectType):\n'
    '    splits = graphene.String()\n'
    '    ablations = graphene.String()\n'
    '    calibration = graphene.String()\n'
    '    robustness = graphene.String()\n'
    '    fairness_harms = graphene.String()\n\n'
    'class MethodsQualMixed(graphene.ObjectType):\n'
    '    sampling = graphene.String()\n'
    '    coding_scheme = graphene.String()\n'
    '    saturation = graphene.String()\n'
    '    triangulation = graphene.String()\n'
    '    reflexivity = graphene.String()\n\n'
    'class MethodsBioHuman(graphene.ObjectType):\n'
    '    approvals_consent = graphene.String()\n'
    '    privacy = graphene.String()\n'
    '    risk_mgmt = graphene.String()\n'
    '    preregistration = graphene.String()\n\n'
    'class MethodsEvidenceAudit(graphene.ObjectType):\n'
    '    design_setup = graphene.String()\n'
    '    statistical_validity = graphene.String()\n'
    '    threats_to_validity = graphene.Field(ThreatsToValidity)\n'
    '    reproducibility = graphene.String()\n'
    '    if_theoretical = graphene.Field(MethodsTheoretical)\n'
    '    if_computational_ml = graphene.Field(MethodsComputational)\n'
    '    if_qualitative_mixed = graphene.Field(MethodsQualMixed)\n'
    '    if_biomedical_human = graphene.Field(MethodsBioHuman)\n\n'
    'class ScoreItem(graphene.ObjectType):\n'
    '    dimension = graphene.String()\n'
    '    score = graphene.Float()\n'
    '    rationale = graphene.String()\n\n'
    'class Readiness(graphene.ObjectType):\n'
    '    status = graphene.String()\n'
    '    justification = graphene.String()\n\n'
    'class LimitationRisk(graphene.ObjectType):\n'
    '    item = graphene.String()\n'
    '    type = graphene.String()\n'
    '    location = graphene.Field(Location)\n\n'
    'class ActionableImprovement(graphene.ObjectType):\n'
    '    priority = graphene.Int()\n'
    '    change = graphene.String()\n'
    '    where = graphene.String()\n'
    '    rationale = graphene.String()\n'
    '    expected_impact = graphene.String()\n\n'
    'class Question(graphene.ObjectType):\n'
    '    question = graphene.String()\n'
    '    location = graphene.Field(Location)\n\n'
    'class Note(graphene.ObjectType):\n'
    '    page = graphene.Int()\n'
    '    anchor = graphene.String()\n'
    '    note = graphene.String()\n\n'
    'class ExecutiveSummary(graphene.ObjectType):\n'
    '    text = graphene.String()\n\n'
    'class NoveltySignificance(graphene.ObjectType):\n'
    '    what_is_new = graphene.String()\n'
    '    missing_comparisons = graphene.List(graphene.String)\n\n'
    'class ClarityOrganization(graphene.ObjectType):\n'
    '    title_abstract_alignment = graphene.String()\n'
    '    structure_flow = graphene.String()\n'
    '    figures_tables_readability = graphene.String()\n'
    '    terminology_consistency = graphene.String()\n'
    '    ambiguities = graphene.String()\n\n'
    'class PageInfo(graphene.ObjectType):\n'
    '    page_no = graphene.Int()\n'
    '    char_len = graphene.Int()\n'
    '    sha256 = graphene.String()\n\n'
    'class Section(graphene.ObjectType):\n'
    '    title = graphene.String()\n'
    '    page_start = graphene.Int()\n'
    '    page_end = graphene.Int()\n'
    '    children = graphene.List(lambda: Section)\n\n'
    'class DocumentStructure(graphene.ObjectType):\n'
    '    pages = graphene.List(PageInfo)\n'
    '    sections = graphene.List(Section)\n\n'
    'class PipelineInfo(graphene.ObjectType):\n'
    '    planner_model = graphene.String()\n'
    '    review_model = graphene.String()\n'
    '    verifier_model = graphene.String()\n'
    '    formatter_model = graphene.String()\n\n'
    'class ProvenanceTimestamps(graphene.ObjectType):\n'
    '    planned_at = graphene.String()\n'
    '    reviewed_at = graphene.String()\n\n'
    'class ProvenanceParams(graphene.ObjectType):\n'
    '    temperature = graphene.Int()\n'
    '    prompt_version = graphene.String()\n'
    '    notes = graphene.String()\n\n'
    'class Provenance(graphene.ObjectType):\n'
    '    pipeline = graphene.Field(PipelineInfo)\n'
    '    timestamps = graphene.Field(ProvenanceTimestamps)\n'
    '    parameters = graphene.Field(ProvenanceParams)\n\n'
    'class Metadata(graphene.ObjectType):\n'
    '    arxiv_id = graphene.String()\n'
    '    title = graphene.String()\n'
    '    authors = graphene.List(graphene.String)\n'
    '    published = graphene.String()\n'
    '    categories = graphene.List(graphene.String)\n'
    '    pdf_sha256 = graphene.String()\n'
    '    license = graphene.String()\n\n'
    'class Review(graphene.ObjectType):\n'
    '    executive_summary = graphene.Field(ExecutiveSummary)\n'
    '    claims_matrix = graphene.Field(ClaimsMatrix)\n'
    '    methods_evidence_audit = graphene.Field(MethodsEvidenceAudit)\n'
    '    novelty_significance = graphene.Field(NoveltySignificance)\n'
    '    clarity_organization = graphene.Field(ClarityOrganization)\n'
    '    limitations_risks = graphene.List(LimitationRisk)\n'
    '    actionable_improvements = graphene.List(ActionableImprovement)\n'
    '    quality_scorecard = graphene.List(ScoreItem)\n'
    '    readiness_level = graphene.Field(Readiness)\n'
    '    questions_to_authors = graphene.List(Question)\n'
    '    line_page_anchored_notes = graphene.List(Note)\n\n'
    'class Assessment(graphene.ObjectType):\n'
    '    metadata = graphene.Field(Metadata)\n'
    '    provenance = graphene.Field(Provenance)\n'
    '    document_structure = graphene.Field(DocumentStructure)\n'
    '    review = graphene.Field(Review)\n'
    '    verification = graphene.JSONString()\n'
    '    arweave = graphene.JSONString()\n\n'
    'class Query(graphene.ObjectType):\n'
    '    paper = graphene.Field(Assessment, arxiv_id=graphene.String(required=True))\n'
    '    all_ids = graphene.List(graphene.String)\n\n'
    '    def resolve_paper(self, info, arxiv_id):\n'
    '        return DATA.get(arxiv_id)\n\n'
    '    def resolve_all_ids(self, info):\n'
    '        return list(DATA.keys())\n\n'
    'schema = graphene.Schema(query=Query)\n'
)

def write_graphql_schema(out_dir: Path) -> Path:
    """Write the generated GraphQL schema into ``schema.py`` within ``out_dir``."""
    safe_mkdir(out_dir)
    schema_path = out_dir / "schema.py"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(GRAPHQL_HEADER)
        f.write("\n")
        f.write(GRAPHQL_SCHEMA)
    return schema_path


# Bundlr / Arweave upload

def bundlr_upload(json_path: Path, wallet: Path, currency: str, tags: List[Dict[str, str]]) -> Optional[str]:
    """Upload a file via the Bundlr CLI. Returns the transaction URL or None on failure."""
    cmd = [
        "bundlr",
        "upload-file",
        str(json_path),
        "-h",
        "https://node1.bundlr.network",
        "-w",
        str(wallet),
        "-c",
        currency,
    ]
    for t in tags:
        cmd.extend(["--tags", f"{t['name']}={t['value']}"])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print("Bundlr CLI not found. Install with: npm install -g @bundlr-network/client")
        return None
    if res.returncode != 0:
        print(f"[bundlr] upload failed: {res.stderr.strip()}")
        return None
    # Try to find an http(s) URL or TX id in output
    tokens = res.stdout.split()
    for tok in tokens:
        if tok.startswith("http://") or tok.startswith("https://"):
            return tok
    return res.stdout.strip() or None


# CLI processing

def process_papers(
    category: str,
    num_papers: int,
    max_results: int,
    output_dir: Path,
    openai_key: Optional[str],
    upload: bool,
    bundlr_wallet: Optional[Path],
    bundlr_currency: str,
    generate_graphql: bool,
    serve_graphql: bool,
    benchmark: bool,
    specific_ids: Optional[List[str]] = None,
    exclude_licenses: Optional[List[str]] = None,
) -> int:
    """Process a set of arXiv papers according to provided CLI options."""
    json_dir = output_dir / "json"
    pdf_dir = output_dir / "pdfs"
    safe_mkdir(json_dir)
    safe_mkdir(pdf_dir)
    # Fetch papers
    metas: List[PaperMetadata] = []
    if specific_ids:
        for pid in specific_ids:
            metas.append(
                PaperMetadata(
                    arxiv_id=pid,
                    title="",
                    abstract="",
                    pdf_url=f"https://arxiv.org/pdf/{pid}.pdf",
                    published="",
                    authors=[],
                    categories=[category] if category else [],
                )
            )
    else:
        feed_xml = fetch_arxiv_feed(category, start=0, max_results=max_results)
        metas_all = parse_arxiv_feed(feed_xml)
        metas = metas_all[: num_papers]
    # OpenAI multi‑model client
    mm: Optional[MultiModel] = None
    if openai_key:
        mm = MultiModel(
            openai_key,
            MODELS["planner"],
            MODELS["reviewer"],
            MODELS["verifier"],
            MODELS["formatter"],
        )
    # Iterate over papers
    for meta in tqdm(metas, desc="Processing papers", unit="paper"):
        # Fetch licence information regardless of CLI flags and store on metadata
        try:
            meta.license = get_arxiv_license(meta.arxiv_id)
        except Exception:
            meta.license = "Unknown"
        pdf_path = pdf_dir / f"{meta.arxiv_id}.pdf"
        try:
            download_pdf(meta.pdf_url, pdf_path)
        except Exception as e:
            print(f"[warn] PDF download failed for {meta.arxiv_id}: {e}")
            continue
        try:
            page_texts, pages_meta = extract_text_per_page(pdf_path)
        except Exception as e:
            print(f"[warn] PDF parse failed for {meta.arxiv_id}: {e}")
            continue
        pdf_hash = sha256_file(pdf_path)
        pages_digest = [(pm.page_no, pm.char_len) for pm in pages_meta]
        planned_at = now_iso()
        plan: Dict[str, Any] = {
            "object": "plan",
            "sections": [],
            "checklist": [],
            "likely_relevant_pages": [],
            "notes": "",
        }
        if mm is not None:
            try:
                plan = mm.plan_checklist(meta.title, meta.abstract, pages_digest)
            except Exception as e:
                print(f"[warn] planning failed for {meta.arxiv_id}: {e}")
        reviewed_at = now_iso()
        assessment: Dict[str, Any] = {}
        if mm is not None:
            try:
                assessment = mm.review_final_json(meta, page_texts, plan, max_tokens=4000)
            except Exception as e:
                print(f"[warn] review generation failed for {meta.arxiv_id}: {e}")
                assessment = {}
        if mm is not None:
            try:
                if not assessment or "review" not in assessment or "metadata" not in assessment:
                    assessment = mm.repair_json_to_schema(assessment or "{}")
            except Exception as e:
                print(f"[warn] formatter repair failed for {meta.arxiv_id}: {e}")
        assessment = ensure_deep_schema_shape(
            obj=assessment or {},
            paper=meta,
            pdf_sha256=pdf_hash,
            pages_meta=pages_meta,
            planner_model=MODELS["planner"],
            reviewer_model=MODELS["reviewer"],
            verifier_model=MODELS["verifier"],
            formatter_model=MODELS["formatter"],
            planned_at=planned_at,
            reviewed_at=reviewed_at,
        )
        if mm is not None:
            try:
                verification = mm.verify_consistency(assessment)
                assessment["verification"] = verification
                assessment["verification"]["json_validated"] = True
            except Exception as e:
                print(f"[warn] verification failed for {meta.arxiv_id}: {e}")
        tags = assessment.get("arweave", {}).get("tags", [])
        base_tags = [
            {"name": "arxiv_id", "value": meta.arxiv_id},
            {"name": "Content-Type", "value": "application/json"},
            {"name": "category", "value": (meta.categories[0] if meta.categories else category)},
            {"name": "app", "value": "arxiv-superreview"},
        ]
        existing = {(t.get("name"), t.get("value")) for t in tags}
        for t in base_tags:
            if (t["name"], t["value"]) not in existing:
                tags.append(t)
        assessment.setdefault("arweave", {})["tags"] = tags
        json_path = json_dir / f"{meta.arxiv_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(assessment, f, ensure_ascii=False, indent=2)
        if upload and bundlr_wallet:
            try:
                tx_url = bundlr_upload(json_path, bundlr_wallet, bundlr_currency, tags)
                if tx_url:
                    print(f"[bundlr] uploaded {meta.arxiv_id}: {tx_url}")
            except Exception as e:
                print(f"[warn] bundlr upload failed for {meta.arxiv_id}: {e}")
        time.sleep(0.5)
    if benchmark:
        chart_path = output_dir / "benchmark.png"
        generate_benchmark_chart(json_dir, chart_path)
        print(f"[ok] benchmark chart saved → {chart_path}")
    if generate_graphql:
        schema_path = write_graphql_schema(output_dir)
        print(f"[ok] GraphQL schema written → {schema_path}")
    if serve_graphql:
        try:
            from flask import Flask  # type: ignore
            from flask_graphql import GraphQLView  # type: ignore
            schema_mod = output_dir / "schema.py"
            if not schema_mod.exists():
                write_graphql_schema(output_dir)
            sys.path.insert(0, str(output_dir))
            import importlib  # type: ignore
            schema_py = importlib.import_module("schema")
            app = Flask(__name__)
            app.add_url_rule(
                "/graphql",
                view_func=GraphQLView.as_view(
                    "graphql", schema=schema_py.schema, graphiql=True
                ),
            )
            print("[ok] Serving GraphQL at http://127.0.0.1:5000/graphql")
            app.run(port=5000)
        except Exception as e:
            print(f"[warn] could not serve GraphQL: {e}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return an argument parser for the CLI."""
    p = argparse.ArgumentParser(
        description="Automated ArXiv Reviewer → Deep JSON → Arweave → GraphQL"
    )
    g_fetch = p.add_argument_group("Fetch")
    g_fetch.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help="arXiv category (e.g., cs.LO)",
    )
    g_fetch.add_argument(
        "--num-papers",
        type=int,
        default=100,
        help="How many most recent papers to process",
    )
    g_fetch.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="How many to fetch before taking the latest num-papers",
    )
    g_fetch.add_argument(
        "--ids",
        nargs="*",
        help="Specific arXiv IDs to process instead of category fetch",
    )
    g_out = p.add_argument_group("Output")
    g_out.add_argument(
        "--output-dir",
        default="out",
        type=Path,
        help="Output directory",
    )
    g_out.add_argument(
        "--benchmark",
        action="store_true",
        help="Generate benchmark charts",
    )
    g_out.add_argument(
        "--generate-graphql",
        action="store_true",
        help="Write a local GraphQL schema.py for the dataset",
    )
    g_out.add_argument(
        "--serve-graphql",
        action="store_true",
        help="Start a local GraphQL HTTP endpoint (requires Flask & flask-graphql)",
    )
    g_openai = p.add_argument_group("OpenAI")
    g_openai.add_argument(
        "--openai-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    g_arweave = p.add_argument_group("Arweave/Bundlr")
    g_arweave.add_argument(
        "--upload",
        action="store_true",
        help="Upload JSON to Arweave via Bundlr",
    )
    g_arweave.add_argument(
        "--bundlr-wallet",
        type=Path,
        help="Path to Bundlr wallet keyfile (JSON)",
    )
    g_arweave.add_argument(
        "--bundlr-currency",
        default="arweave",
        help="Bundlr currency (arweave|matic|solana|... )",
    )

    # License handling (placeholder)
    # This flag accepts one or more license names to exclude.  The
    # underlying implementation currently always fetches the license and
    # stores it in the JSON and GraphQL regardless of this flag.  In
    # future versions the list could be used to filter licences.
    p.add_argument(
        "--exclude-licenses",
        nargs="*",
        default=[],
        help="Exclude specified license types from processing (currently unused, placeholder)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the CLI."""
    args = build_arg_parser().parse_args(argv)
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY", "")
    if args.upload and not args.bundlr_wallet:
        print("--upload requires --bundlr-wallet", file=sys.stderr)
        return 2
    return process_papers(
        category=args.category,
        num_papers=args.num_papers,
        max_results=args.max_results,
        output_dir=args.output_dir,
        openai_key=openai_key if openai_key else None,
        upload=bool(args.upload),
        bundlr_wallet=args.bundlr_wallet,
        bundlr_currency=args.bundlr_currency,
        generate_graphql=bool(args.generate_graphql),
        serve_graphql=bool(args.serve_graphql),
        benchmark=bool(args.benchmark),
        specific_ids=args.ids if args.ids else None,
        exclude_licenses=args.exclude_licenses,
    )


if __name__ == "__main__":
    sys.exit(main())