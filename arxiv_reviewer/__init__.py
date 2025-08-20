#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated ArXiv Reviewer → Deep JSON → Arweave (Bundlr) → GraphQL

This module implements a pipeline that can fetch papers from arXiv, extract
their contents, generate structured reviews using OpenAI models, optionally
upload the results to Arweave via Bundlr, produce benchmark charts, and emit
a GraphQL schema for local consumption.  The emphasis is on producing a
deeply nested JSON representation of each paper with rich provenance
metadata.  This version contains fixes for planner and verification
failures and removes a naive summarisation fallback.

Features:
  • Fetch the latest arXiv preprints for a given category (default
    ``cs.LO``) or process specific arXiv IDs.  A ``--date`` option allows
    restricting the search to a specific submission date or an inclusive
    date range.
  • Download PDFs and extract text page‑by‑page via PyMuPDF (``fitz``), with
    a fallback to PyPDF2 if unavailable.
  • Use a multi‑model strategy with OpenAI to plan, review, verify and
    format the assessment into a strict JSON schema.  The planner and
    verifier have been updated to request JSON mode when supported and
    include improved prompts.
  • Persist each assessment as a JSON file and, optionally, upload it to
    Arweave via Bundlr with appropriate metadata tags.
  • Optionally generate benchmark charts summarising readiness levels and
    average scores across a corpus of assessments.
  • Optionally produce a GraphQL schema and run a local GraphQL endpoint to
    explore the generated data.

Notes:
  • Only the reviewer model (e.g. ``gpt-5``) performs any evaluation; the
    planner, verifier and formatter models are non‑evaluative.
  • When generating charts, the script strives to avoid clutter by using
    consistent fonts, legible labels and high DPI settings.
  • The JSON repair pass with the formatter model is non‑evaluative and
    exists solely to enforce schema compliance.
  • This version requests the ``response_format`` parameter where supported
    to encourage models to return syntactically valid JSON.  If a model
    rejects this parameter the helper will disable it for future calls【257285575724731†L18-L22】.

Disclaimer:
    This script calls external services (arXiv, OpenAI, Bundlr).  Make sure
    you have appropriate API keys and respect applicable rate limits and
    terms of use.
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

# Configuration defaults
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

# Number of layers applied to quality scores to temper scoring generosity
SCORE_LAYERS: int = int(os.getenv("SCORE_LAYERS", "2"))

__all__ = [
    "process_papers",
    "build_arg_parser",
    "main",
    "MODELS",
    "SCORE_LAYERS",
]

# Aesthetic defaults for charts
PLOT_DPI: int = 180
PLOT_STYLE: str = "whitegrid"
PLOT_FONT_SIZE: int = 10

# Default structure for an empty review.  When model calls fail or
# return invalid JSON, the pipeline will still produce a review object
# with the correct keys but empty values.  This prevents downstream
# consumers from encountering missing fields.
DEFAULT_REVIEW: Dict[str, Any] = {
    "executive_summary": {"text": ""},
    "claims_matrix": {"columns": [], "rows": []},
    "methods_evidence_audit": {
        "design_setup": "",
        "statistical_validity": "",
        "threats_to_validity": {
            "internal": "",
            "external": "",
            "construct": "",
        },
        "reproducibility": "",
        "if_theoretical": {
            "definitions": "",
            "lemmas": "",
            "proof_gaps": "",
            "boundary_cases": "",
            "counterexamples": "",
        },
        "if_computational_ml": {
            "splits": "",
            "ablations": "",
            "calibration": "",
            "robustness": "",
            "fairness_harms": "",
        },
        "if_qualitative_mixed": {
            "sampling": "",
            "coding_scheme": "",
            "saturation": "",
            "triangulation": "",
            "reflexivity": "",
        },
        "if_biomedical_human": {
            "approvals_consent": "",
            "privacy": "",
            "risk_mgmt": "",
            "preregistration": "",
        },
    },
    "novelty_significance": {
        "what_is_new": "",
        "missing_comparisons": [],
    },
    "clarity_organization": {
        "title_abstract_alignment": "",
        "structure_flow": "",
        "figures_tables_readability": "",
        "terminology_consistency": "",
        "ambiguities": "",
    },
    "limitations_risks": [],
    "actionable_improvements": [],
    "quality_scorecard": [],
    "readiness_level": {
        "status": "",
        "justification": "",
    },
    "questions_to_authors": [],
    "line_page_anchored_notes": [],
}


@dataclass
class PaperMetadata:
    """Metadata describing a paper fetched from arXiv."""
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str
    published: str
    authors: List[str]
    categories: List[str]
    license: str = ""
    msc_classes: List[str] = dataclasses.field(default_factory=list)
    acm_classes: List[str] = dataclasses.field(default_factory=list)


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
    return dt.datetime.now(dt.timezone.utc).isoformat()


def shorten(s: str, n: int = 80) -> str:
    """Shorten a string to ``n`` characters, replacing newlines with spaces."""
    s = s.replace("\n", " ")
    return (s[: n - 1] + "…") if len(s) > n else s


def sleep_backoff(tries: int) -> None:
    """Sleep with exponential backoff up to 10 seconds."""
    time.sleep(min(10, 1.5 ** tries))


def _parse_date_range(date_str: str) -> Tuple[str, str]:
    """Parse a date or date range string into arXiv API format (YYYYMMDDHHMM)."""
    parts = [p.strip() for p in date_str.split(":", 1)]
    if not parts or not parts[0]:
        raise ValueError("Invalid date specification")
    if len(parts) == 1:
        start_part = end_part = parts[0]
    else:
        start_part, end_part = parts

    def to_yyyymmddhhmm(s: str, end: bool = False) -> str:
        try:
            dt_obj = dt.datetime.strptime(s, "%Y-%m-%d")
        except Exception as exc:
            raise ValueError(f"Invalid date format '{s}'. Expected YYYY-MM-DD") from exc
        ymd = dt_obj.strftime("%Y%m%d")
        return f"{ymd}{'2359' if end else '0000'}"

    return to_yyyymmddhhmm(start_part), to_yyyymmddhhmm(end_part, end=True)


def fetch_arxiv_feed(category: str, start: int, max_results: int, *, date_range: Optional[str] = None) -> str:
    """Fetch an Atom feed from arXiv for a given category, optionally filtering by submission date."""
    if date_range:
        start_str, end_str = _parse_date_range(date_range)
        search_query = f"cat:{category} AND submittedDate:[{start_str} TO {end_str}]"
    else:
        search_query = f"cat:{category}"
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
    }
    r = requests.get(ARXIV_API, params=params, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    return r.text


def fetch_arxiv_entry_by_id(arxiv_id: str) -> Optional[PaperMetadata]:
    """Fetch a single arXiv entry by its identifier using the arXiv API."""
    try:
        resp = requests.get(
            ARXIV_API,
            params={"id_list": arxiv_id},
            headers={"User-Agent": UA},
            timeout=60,
        )
        resp.raise_for_status()
        entries = parse_arxiv_feed(resp.text)
        return entries[0] if entries else None
    except Exception:
        return None


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
        cats_raw: List[str] = [c.attrib.get("term", "") for c in e.findall("atom:category", ns)]
        cats: List[str] = []
        msc_classes: List[str] = []
        acm_classes: List[str] = []
        for term in cats_raw:
            term = term.strip()
            if ";" in term:
                acm_classes.extend([t.strip() for t in term.split(";") if t.strip()])
            elif "." in term and term.split(".")[0].isalpha():
                cats.append(term)
            elif any(ch.isdigit() for ch in term):
                msc_classes.extend([t.strip() for t in term.split(",") if t.strip()])
            else:
                cats.append(term)
        pdf_url = ""
        for link in e.findall("atom:link", ns):
            t = link.attrib.get("type", "")
            title_attr = link.attrib.get("title", "")
            if t == "application/pdf" or (title_attr and title_attr.lower() == "pdf"):
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
                msc_classes=msc_classes,
                acm_classes=acm_classes,
            )
        )
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
    """Fetch the license associated with an arXiv paper by scraping the abstract page."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        html = r.text
        m = re.search(r'abs-license.*?href="([^"]+)"', html, re.DOTALL)
        if m:
            lic_url = m.group(1)
            cc = re.search(r'/licenses/([^/]+)/([0-9.]+)/', lic_url)
            if cc:
                code, version = cc.group(1), cc.group(2)
                code_name = code.replace('-', '-').upper()
                return f"CC {code_name} {version}"
            last = lic_url.rstrip('/').split('/')[-1]
            return last.replace('-', ' ').upper()
        return "Unknown"
    except Exception:
        return "Unknown"


def extract_text_per_page(pdf_path: Path) -> Tuple[List[str], List[PageInfo]]:
    """Extract text from a PDF page by page."""
    pages_text: List[str] = []
    pages_meta: List[PageInfo] = []
    if HAVE_FITZ:
        try:
            with fitz.open(pdf_path) as doc:
                for i, page in enumerate(doc):
                    txt = page.get_text("text")
                    b = txt.encode("utf-8", errors="ignore")
                    pages_text.append(txt)
                    pages_meta.append(PageInfo(page_no=i + 1, char_len=len(b), sha256=sha256_bytes(b)))
            return pages_text, pages_meta
        except Exception:
            pass
    if HAVE_PYPDF2:
        with open(pdf_path, "rb") as fh:
            pdf_reader = PyPDF2.PdfReader(fh)
            for i, page in enumerate(pdf_reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                b = txt.encode("utf-8", errors="ignore")
                pages_text.append(txt)
                pages_meta.append(PageInfo(page_no=i + 1, char_len=len(b), sha256=sha256_bytes(b)))
        return pages_text, pages_meta
    raise RuntimeError("Neither PyMuPDF nor PyPDF2 is available to extract text from PDF")


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
    score_layers: int,
) -> Dict[str, Any]:
    """Ensure that a deep assessment JSON has the correct top-level structure."""
    assessment: Dict[str, Any] = dict(obj) if obj is not None else {}
    assessment.setdefault("metadata", {})
    assessment.setdefault("provenance", {})
    assessment.setdefault("document_structure", {})
    assessment.setdefault("review", {})
    assessment.setdefault("verification", {})
    assessment.setdefault("arweave", {})
    assessment["metadata"]["arxiv_id"] = paper.arxiv_id
    assessment["metadata"]["title"] = paper.title
    assessment["metadata"]["authors"] = paper.authors
    assessment["metadata"]["published"] = paper.published
    assessment["metadata"]["categories"] = paper.categories
    assessment["metadata"]["pdf_sha256"] = pdf_sha256
    assessment["metadata"]["license"] = paper.license
    assessment["metadata"]["msc_classes"] = getattr(paper, "msc_classes", [])
    assessment["metadata"]["acm_classes"] = getattr(paper, "acm_classes", [])
    assessment["provenance"].setdefault("pipeline", {})
    assessment["provenance"]["pipeline"]["planner_model"] = planner_model
    assessment["provenance"]["pipeline"]["review_model"] = reviewer_model
    assessment["provenance"]["pipeline"]["verifier_model"] = verifier_model
    assessment["provenance"]["pipeline"]["formatter_model"] = formatter_model
    assessment["provenance"].setdefault("timestamps", {})
    assessment["provenance"]["timestamps"]["planned_at"] = planned_at
    assessment["provenance"]["timestamps"]["reviewed_at"] = reviewed_at
    assessment["provenance"].setdefault("parameters", {})
    assessment["provenance"]["parameters"].setdefault("temperature", 0)
    assessment["provenance"]["parameters"].setdefault("prompt_version", "v1")
    assessment["provenance"]["parameters"].setdefault("notes", "")
    assessment["provenance"]["parameters"]["score_layers"] = score_layers
    assessment["document_structure"].setdefault("pages", [])
    assessment["document_structure"]["pages"] = [
        {"page_no": p.page_no, "char_len": p.char_len, "sha256": p.sha256} for p in pages_meta
    ]
    review = assessment.setdefault("review", {})
    for k, v in DEFAULT_REVIEW.items():
        if k not in review:
            review[k] = json.loads(json.dumps(v))
        else:
            if isinstance(v, dict) and isinstance(review[k], dict):
                def merge_defaults(src: Dict[str, Any], dest: Dict[str, Any]) -> None:
                    for kk, vv in src.items():
                        if kk not in dest:
                            dest[kk] = json.loads(json.dumps(vv)) if isinstance(vv, (dict, list)) else vv
                        elif isinstance(vv, dict) and isinstance(dest[kk], dict):
                            merge_defaults(vv, dest[kk])
                merge_defaults(v, review[k])
    return assessment


def apply_score_layers(assessment: Dict[str, Any], layers: int) -> None:
    """Apply additional layers to dampen generous quality scores."""
    if layers <= 1:
        return
    try:
        scorecard = assessment.get("review", {}).get("quality_scorecard", [])
        for item in scorecard:
            score = float(item.get("score", 0.0)) / layers
            item["score"] = max(0.0, min(1.0, score))
    except Exception:
        pass


class MultiModel:
    """
    Wrapper around OpenAI models to coordinate planning, reviewing, verifying and
    formatting.  Each method delegates to the appropriate model and catches
    common exceptions to provide diagnostics.  When no API key is provided or
    the OpenAI library is unavailable, construction of this class will raise
    a RuntimeError.
    """
    def __init__(self, openai_key: str, planner_model: str, reviewer_model: str, verifier_model: str, formatter_model: str) -> None:
        if not HAVE_OPENAI:
            raise RuntimeError("openai package is not installed")
        self.client = OpenAI(api_key=openai_key)
        self.models: Dict[str, str] = {
            "planner": planner_model,
            "reviewer": reviewer_model,
            "verifier": verifier_model,
            "formatter": formatter_model,
        }
        self.temperature_supported: bool = True
        self.max_tokens_supported: bool = True
        self.response_format_supported: bool = True
        self._max_tokens_warning_printed = False
        self._temperature_warning_printed = False

    def plan_checklist(self, title: str, abstract: str, pages_digest: Iterable[Tuple[int, int]]) -> Dict[str, Any]:
        """Use the planner model to produce a review plan and evidence checklist."""
        try:
            pages_list = [{"page": p, "len": l} for (p, l) in pages_digest]
        except Exception:
            pages_list = []
        msg_user = (
            "You are preparing a plan for reviewing a research paper.\n"
            "Return STRICT JSON (no prose) with keys: {\"object\":\"plan\",\"sections\":[{\"name\":\"...\",\"what_to_look_for\":[...] }],\"checklist\":[\"...\"],\"likely_relevant_pages\":[{\"page\":int,\"priority\":\"high|medium|low\"}],\"notes\":\"short\"}.\n"
            "Use ONLY the abstract/title and page lengths. No evaluation.\n\n"
            f"TITLE: {title}\n\nABSTRACT:\n{abstract}\n\n"
            "PAGES DIGEST (page_no, char_count):\n"
            + json.dumps(pages_list, ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": "You are a meticulous planning assistant. Respond in strict JSON only."},
            {"role": "user", "content": msg_user},
        ]
        tries = 0
        while True:
            try:
                resp = self._chat_create(
                    model_key="planner",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=900,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                with contextlib.suppress(Exception):
                    return json_loads_strict(content)
                return {"object": "plan", "sections": [], "checklist": [], "likely_relevant_pages": [], "notes": ""}
            except Exception:
                tries += 1
                if tries > 3:
                    return {"object": "plan", "sections": [], "checklist": [], "likely_relevant_pages": [], "notes": ""}
                sleep_backoff(tries)

    def review_final_json(self, meta: PaperMetadata, page_texts: List[str], plan: Dict[str, Any], max_tokens: int = 4000) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a reviewer for academic papers. Produce JSON strictly following the plan."},
            {"role": "user", "content": json.dumps({"metadata": dataclasses.asdict(meta), "plan": plan, "pages": page_texts})},
        ]
        tries = 0
        while True:
            try:
                resp = self._chat_create(
                    model_key="reviewer",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                return json_loads_strict(content)
            except Exception:
                tries += 1
                if tries > 3:
                    raise
                sleep_backoff(tries)

    def verify_consistency(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the verifier model to point out internal consistency issues."""
        instr = (
            "You are a verifier. Inspect the provided JSON assessment for internal consistency ONLY.\n"
            "Return STRICT JSON with keys: {\"object\":\"verification\",\"issues\":[{\"field\":\"path\",\"problem\":\"...\",\"severity\":\"low|medium|high\"}],\"notes\":\"short\"}.\n"
            "Do NOT re-evaluate the paper. Do NOT modify the assessment."
        )
        tries = 0
        while True:
            try:
                resp = self._chat_create(
                    model_key="verifier",
                    messages=[
                        {"role": "system", "content": instr},
                        {"role": "user", "content": json.dumps(assessment, ensure_ascii=False)},
                    ],
                    temperature=0.0,
                    max_tokens=800,
                    reasoning_effort="medium",
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                return json_loads_strict(content)
            except Exception:
                tries += 1
                if tries >= 2:
                    return {"object": "verification", "issues": [], "notes": "verification skipped after retries"}
                sleep_backoff(tries)

    def repair_json_to_schema(self, assessment_json: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Request the formatter model to repair a potentially invalid JSON object and
        ensure it conforms to the expected schema.  If the formatter model
        returns an empty string or non‑JSON content, this method substitutes an
        empty JSON object and raises a more informative exception containing a
        snippet of the offending content.
        """
        if isinstance(assessment_json, dict):
            content_str = json.dumps(assessment_json)
        else:
            content_str = assessment_json
        messages = [
            {"role": "system", "content": "You are a formatter ensuring that the JSON complies with the expected schema."},
            {"role": "user", "content": content_str},
        ]
        tries = 0
        while True:
            try:
                resp = self._chat_create(
                    model_key="formatter",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                if not isinstance(content, str) or not content.strip():
                    content = "{}"
                try:
                    return json_loads_strict(content)
                except Exception as parse_exc:
                    snippet = content[:200] + ("..." if len(content) > 200 else "")
                    raise ValueError(
                        f"Formatter JSON parse failed: {parse_exc}; content snippet: {snippet}"
                    ) from parse_exc
            except Exception:
                tries += 1
                if tries > 3:
                    raise
                sleep_backoff(tries)

    def _chat_create(self,
                     model_key: str,
                     messages: List[Dict[str, Any]],
                     temperature: float = 0.0,
                     max_tokens: Optional[int] = None,
                     **extra: Any) -> Any:
        """Helper to call the OpenAI chat API with optional ``max_tokens`` and ``response_format`` handling."""
        params: Dict[str, Any] = {
            "model": self.models[model_key],
            "messages": messages,
        }
        if self.temperature_supported:
            params["temperature"] = temperature
        if self.max_tokens_supported and max_tokens is not None:
            params["max_tokens"] = max_tokens
        # Pull response_format out of extra if present
        if "response_format" in extra and self.response_format_supported:
            params["response_format"] = extra.pop("response_format")
        # Merge any remaining extra options
        params.update(extra)
        try:
            return self.client.chat.completions.create(**params)
        except Exception as e:
            msg = str(e)
            # Unsupported response_format
            if self.response_format_supported and "response_format" in msg and ("unsupported" in msg or "not supported" in msg or "does not support" in msg):
                self.response_format_supported = False
                params.pop("response_format", None)
                print('[warn] The "response_format" parameter is not supported by the selected model; disabling it for future calls')
                return self.client.chat.completions.create(**params)
            # Unsupported max_tokens
            if self.max_tokens_supported and "max_tokens" in msg and ("unsupported" in msg or "not supported" in msg or "does not support" in msg):
                self.max_tokens_supported = False
                if not self._max_tokens_warning_printed:
                    print('[warn] The "max_tokens" parameter is not supported by the selected model; disabling it for future calls')
                    self._max_tokens_warning_printed = True
                params.pop("max_tokens", None)
                return self.client.chat.completions.create(**params)
            # Unsupported temperature
            if self.temperature_supported and "temperature" in msg and ("unsupported" in msg or "does not support" in msg):
                self.temperature_supported = False
                if not self._temperature_warning_printed:
                    print('[warn] The specified "temperature" is not supported by the selected model; using the default value for future calls')
                    self._temperature_warning_printed = True
                params.pop("temperature", None)
                return self.client.chat.completions.create(**params)
            # Propagate other errors
            raise


def bundlr_upload(json_path: Path, wallet_path: Path, currency: str, tags: List[Dict[str, str]]) -> Optional[str]:
    """Upload a JSON file to Arweave using the Bundlr CLI."""
    try:
        cmd = [
            "bundlr",
            "upload",
            str(json_path),
            "--wallet",
            str(wallet_path),
            "--currency",
            currency,
            "--tags",
            json.dumps(tags),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        m = re.search(r"https?://[\w./-]+", result.stdout)
        return m.group(0) if m else None
    except Exception as e:
        print(f"[warn] bundlr upload error: {e}")
        return None


def generate_benchmark_chart(json_dir: Path, out_path: Path) -> None:
    """Generate a benchmark chart summarising quality scores and readiness levels."""
    if sns is None or not HAVE_MPL:
        print("[warn] seaborn/matplotlib not available; cannot generate benchmark chart")
        return
    items: List[Tuple[str, float]] = []
    readiness: List[Tuple[str, str]] = []
    for fp in json_dir.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            scores = obj.get("review", {}).get("quality_scorecard", [])
            for itm in scores:
                items.append((itm.get("dimension", ""), itm.get("score", 0.0)))
            rl = obj.get("review", {}).get("readiness_level", {})
            readiness.append((fp.stem, rl.get("status", "Unknown")))
        except Exception:
            continue
    if not items:
        print("[warn] no scorecard data to plot")
        return
    dims = [i[0] for i in items]
    scores = [i[1] for i in items]
    plt.figure(dpi=PLOT_DPI)
    sns.barplot(x=dims, y=scores)
    plt.xticks(rotation=45, ha="right", fontsize=PLOT_FONT_SIZE)
    plt.yticks(fontsize=PLOT_FONT_SIZE)
    plt.title("Quality Scorecard Averages")
    plt.tight_layout()
    plt.savefig(out_path)


def write_graphql_schema(output_dir: Path) -> Path:
    """Emit a GraphQL schema into ``schema.py`` within the provided ``output_dir``."""
    schema_src = Path(__file__).with_name("schema.py").read_text()
    dest = output_dir / "schema.py"
    dest.write_text(schema_src)
    return dest


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
    score_layers: int = SCORE_LAYERS,
    specific_ids: Optional[List[str]] = None,
    exclude_licenses: Optional[List[str]] = None,
    date: Optional[str] = None,
) -> int:
    """Process a set of arXiv papers according to provided CLI options."""
    json_dir = output_dir / "json"
    pdf_dir = output_dir / "pdfs"
    safe_mkdir(json_dir)
    safe_mkdir(pdf_dir)
    metas: List[PaperMetadata] = []
    if specific_ids:
        for pid in specific_ids:
            entry = fetch_arxiv_entry_by_id(pid)
            if entry is not None:
                metas.append(entry)
            else:
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
        feed_xml = fetch_arxiv_feed(category, start=0, max_results=max_results, date_range=date)
        metas_all = parse_arxiv_feed(feed_xml)
        metas_all = [m for m in metas_all if '/' not in m.arxiv_id]
        metas = metas_all[: num_papers]
    mm: Optional[MultiModel] = None
    if openai_key:
        mm = MultiModel(
            openai_key,
            MODELS["planner"],
            MODELS["reviewer"],
            MODELS["verifier"],
            MODELS["formatter"],
        )
    for meta in tqdm(metas, desc="Processing papers", unit="paper"):
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
        fallback_summary = meta.abstract.strip() if meta.abstract else ""
        pdf_hash = sha256_file(pdf_path)
        pages_digest = [(pm.page_no, pm.char_len) for pm in pages_meta]
        planned_at = now_iso()
        plan: Dict[str, Any] = {"object": "plan", "sections": [], "checklist": [], "likely_relevant_pages": [], "notes": ""}
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
                try:
                    assessment_snippet = json.dumps(assessment)[:200] if isinstance(assessment, dict) else str(assessment)[:200]
                except Exception:
                    assessment_snippet = repr(assessment)[:200]
                print(f"[warn] formatter repair failed for {meta.arxiv_id}: {e}; assessment snippet: {assessment_snippet}")
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
            score_layers=score_layers,
        )
        apply_score_layers(assessment, score_layers)
        if mm is not None:
            try:
                verification = mm.verify_consistency(assessment)
                assessment["verification"] = verification
                assessment["verification"]["json_validated"] = True
            except Exception as e:
                print(f"[warn] verification failed for {meta.arxiv_id}: {e}")
        try:
            current_summary = assessment.get("review", {}).get("executive_summary", {}).get("text", "")
            if not current_summary and fallback_summary:
                assessment["review"]["executive_summary"]["text"] = fallback_summary
        except Exception:
            pass
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
    g_fetch.add_argument(
        "--date",
        default=None,
        help=(
            "Submission date or range to restrict search to. Accepts YYYY-MM-DD or "
            "YYYY-MM-DD:YYYY-MM-DD. If omitted, the latest papers are returned."
        ),
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
    g_openai.add_argument(
        "--planner-model",
        default=MODELS["planner"],
        help="Model to use for planning",
    )
    g_openai.add_argument(
        "--reviewer-model",
        default=MODELS["reviewer"],
        help="Model to use for reviewing",
    )
    g_openai.add_argument(
        "--verifier-model",
        default=MODELS["verifier"],
        help="Model to use for verification",
    )
    g_openai.add_argument(
        "--formatter-model",
        default=MODELS["formatter"],
        help="Model to use for JSON repair",
    )
    g_eval = p.add_argument_group("Scoring")
    g_eval.add_argument(
        "--score-layers",
        type=int,
        default=SCORE_LAYERS,
        help="Number of layers to temper quality scores",
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
    p.add_argument(
        "--exclude-licenses",
        nargs="*",
        default=[],
        help="Exclude specified license types from processing (currently unused)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the CLI."""
    args = build_arg_parser().parse_args(argv)
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY", "")
    if args.upload and not args.bundlr_wallet:
        print("--upload requires --bundlr-wallet", file=sys.stderr)
        return 2
    MODELS["planner"] = args.planner_model
    MODELS["reviewer"] = args.reviewer_model
    MODELS["verifier"] = args.verifier_model
    MODELS["formatter"] = args.formatter_model
    global SCORE_LAYERS
    SCORE_LAYERS = args.score_layers
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
        score_layers=SCORE_LAYERS,
        specific_ids=args.ids if args.ids else None,
        exclude_licenses=args.exclude_licenses,
        date=args.date,
    )


if __name__ == "__main__":
    sys.exit(main())
