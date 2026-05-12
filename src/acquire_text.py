"""Stage 2 - acquire the text payload appropriate to each source.

Different source types contribute very different amounts of text. Treating them
uniformly would let long technical specifications dominate any similarity
measure simply by length. We therefore use the text-acquisition strategy
encoded in `sources.csv:text_strategy` (set by src/parse_references.py from the
rules in docs/methodology.md §2).

    abstract  -> abstract + title + keywords (academic, survey, preprint)
    summary   -> executive summary, foreword, or first ~1,500 words
                 (regulatory guidance, technical white papers, standards)
    full      -> entire article text (short industry blogs)
    manual    -> requires a human to paste a chosen excerpt
                 (software repositories, items with no detectable type)

This module is deliberately conservative. It will:

  * Use CrossRef for DOI-bearing sources to fetch abstracts.
  * Use the arXiv API for arXiv preprints.
  * Read a local PDF (when the user has placed one in data/abstracts_cache/)
    and extract the first N words.
  * Read a local .txt override at data/abstracts_cache/{id}.txt if present,
    which always takes precedence. This is how human-curated payloads enter
    the pipeline.

It will NOT scrape arbitrary websites, because the legality and reliability of
doing so vary by publisher. Where automated retrieval fails the row is left
empty and surfaced in the curation report (see docs/topic_assignment_guide.md).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import pandas as pd
import requests


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "abstracts_cache"
SUMMARY_WORD_BUDGET = 1500
USER_AGENT = (
    "survey-source-visualisation/0.1 "
    "(academic survey; contact: holly.baker@newtoneurope.com)"
)


def _override_path(source_id: str) -> Path:
    return CACHE_DIR / f"{source_id}.txt"


def _cache_path(source_id: str) -> Path:
    return CACHE_DIR / f"{source_id}.json"


def _read_override(source_id: str) -> Optional[str]:
    p = _override_path(source_id)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def _read_cache(source_id: str) -> Optional[dict]:
    p = _cache_path(source_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _write_cache(source_id: str, payload: dict) -> None:
    _cache_path(source_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_crossref_abstract(doi: str, session: requests.Session) -> Optional[str]:
    """Fetch an abstract from CrossRef. JATS-XML markup is stripped naively."""
    if not doi:
        return None
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = session.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
    except requests.RequestException:
        return None
    data = r.json().get("message", {})
    abstract = data.get("abstract")
    if not abstract:
        return None
    # Strip JATS tags and entities.
    abstract = re.sub(r"<[^>]+>", " ", abstract)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract or None


def fetch_arxiv_abstract(arxiv_id: str, session: requests.Session) -> Optional[str]:
    if not arxiv_id:
        return None
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        r = session.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
    except requests.RequestException:
        return None
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return None
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    summary_el = root.find(".//atom:entry/atom:summary", ns)
    if summary_el is None or summary_el.text is None:
        return None
    return re.sub(r"\s+", " ", summary_el.text).strip()


def truncate_words(text: str, budget: int) -> str:
    words = text.split()
    return " ".join(words[:budget])


def extract_pdf_summary(pdf_path: Path, budget: int = SUMMARY_WORD_BUDGET) -> str:
    """Extract roughly `budget` words from the start of a PDF.

    We import pdfplumber lazily because it pulls in a large dependency stack.
    """
    import pdfplumber  # type: ignore

    out: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            out.append(page_text)
            if sum(len(t.split()) for t in out) >= budget:
                break
    text = " ".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return truncate_words(text, budget)


def acquire_for_row(row: pd.Series, session: requests.Session) -> dict:
    """Resolve text content for a single source row.

    Returns a dict with keys `text`, `source` (where it came from), and
    `strategy` (the strategy actually applied, which may differ from the
    seeded strategy if a manual override was supplied).
    """
    source_id = row["id"]

    override = _read_override(source_id)
    if override is not None:
        return {"text": override, "source": "override", "strategy": "manual"}

    cached = _read_cache(source_id)
    if cached and cached.get("text"):
        return cached

    strategy = row.get("text_strategy") or "manual"
    payload: dict = {"text": "", "source": "", "strategy": strategy}

    if strategy == "abstract":
        text = fetch_crossref_abstract(str(row.get("doi", "")), session)
        if text:
            payload.update(text=text, source="crossref")
        else:
            text = fetch_arxiv_abstract(str(row.get("arxiv_id", "")), session)
            if text:
                payload.update(text=text, source="arxiv")
    elif strategy == "summary":
        pdf = CACHE_DIR / f"{source_id}.pdf"
        if pdf.exists():
            payload.update(
                text=extract_pdf_summary(pdf),
                source=f"local_pdf:{pdf.name}",
            )
    elif strategy == "full":
        # Full-text retrieval for arbitrary blog hosts is intentionally
        # out of scope; place the article body at data/abstracts_cache/{id}.txt.
        payload.update(strategy="full")
    else:
        payload.update(strategy="manual")

    if payload["text"]:
        _write_cache(source_id, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", default="data/sources.csv")
    ap.add_argument("--out", default="data/source_text.csv")
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="pause between network calls to be polite to CrossRef/arXiv",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / args.sources)
    session = requests.Session()

    rows = []
    for _, row in df.iterrows():
        result = acquire_for_row(row, session)
        rows.append(
            {
                "id": row["id"],
                "strategy_applied": result["strategy"],
                "source": result["source"],
                "text": result["text"],
            }
        )
        if result["source"] in {"crossref", "arxiv"}:
            time.sleep(args.sleep)

    out_path = root / args.out
    pd.DataFrame(rows).to_csv(out_path, index=False)
    have = sum(1 for r in rows if r["text"])
    print(f"resolved text for {have}/{len(rows)} sources -> {args.out}")


if __name__ == "__main__":
    main()
