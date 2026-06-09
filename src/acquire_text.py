"""Stage 2 - acquire the text payload appropriate to each source.

The semantic clusters in the figure are built from this text, so what we feed
in per source matters more than anything downstream. Different source types
carry their meaning in different places, and treating them uniformly would let
a long document dominate a short one by length alone. We therefore acquire a
**type-appropriate, length-controlled** payload, driven by the
`text_strategy` column that `parse_references.py` derives from publication type:

    Strategy   Source types                              What we acquire and why
    --------   ----------------------------------------  ----------------------------------------
    abstract   academic_paper, preprint, survey_review   The author-written abstract - a curated
                                                          semantic summary of the contribution.
                                                          arXiv API, then Crossref, then Semantic
                                                          Scholar, then the landing page.
    summary    regulatory_guidance, technical_whitepaper,The document's own description / scope
               standards_specification                   statement: the page meta-description and
                                                          opening paragraphs (first ~1,500 chars).
                                                          The opening states intent for these long,
                                                          narrative documents.
    full       industry_blog                             The page description + lead paragraphs;
                                                          blogs are already short and scoped.
    manual     online_resource, software_repository      The landing-page / repository description
                                                          (meta-description, README lead): the
                                                          "high-level description equivalent" for
                                                          things that have no abstract.

Modelling assumptions, stated plainly so they can be contested:

  1. The abstract (academic) or the self-description (everything else) is a
     faithful, lower-variance proxy for what a source is *about*. We are
     mapping stated scope, not full content.
  2. The title is always informative, so the clustering text is the title
     joined to the acquired body (see build_figures). A source therefore never
     has empty text: if acquisition fails, it falls back to title (+ the venue
     lead for untitled reports), and that fallback is recorded explicitly in
     `source` so coverage can be audited.
  3. Length is controlled by capping the body, so a 300-page standard and a
     4-page paper enter on comparable footing.

Provenance for every source is written to `data/source_text.csv` (one row per
source, with `source`, `status`, `http_status`, `n_chars`, `title_seen`) so a
reader can verify that every abstract / equivalent was considered and see
exactly where each came from. A human-supplied override at
`data/abstracts_cache/{id}.txt` always wins.

Implementation note: this uses only the Python standard library (urllib) so it
runs in a minimal environment. It fetches public bibliographic APIs (arXiv,
Crossref, Semantic Scholar) and public landing pages; it never invents text.
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "abstracts_cache"
WEB_CHAR_BUDGET = 1500
ABSTRACT_CHAR_BUDGET = 3000
MAX_BYTES = 600_000
RETRY_CODES = {429, 500, 502, 503, 504}
CTX = ssl.create_default_context()
# APIs (Crossref/arXiv/Semantic Scholar) prefer a contactable identifier; many
# publisher and government landing pages reject non-browser agents, so web
# pages are fetched with a browser UA instead.
API_UA = (
    "survey-source-visualisation/0.2 "
    "(academic survey; contact: holly.baker@newtoneurope.com)"
)
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
PDF_MAX_BYTES = 8_000_000


# --------------------------------------------------------------------------- #
# low-level fetch
# --------------------------------------------------------------------------- #
def _fetch(url: str, accept: str | None = None, timeout: int = 20, retries: int = 3,
           ua: str = API_UA, max_bytes: int = MAX_BYTES):
    """Return (http_status, content_type, raw_bytes). Never raises."""
    headers = {"User-Agent": ua}
    if accept:
        headers["Accept"] = accept
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return r.status, r.headers.get_content_type(), r.read(max_bytes)
        except urllib.error.HTTPError as e:
            if e.code in RETRY_CODES and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            return e.code, None, b""
        except Exception:  # noqa: BLE001 - URLError, timeout, ssl, etc.
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, None, b""
    return None, None, b""


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


def _clean_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = unescape(s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# bibliographic-API extractors
# --------------------------------------------------------------------------- #
def fetch_arxiv(arxiv_id: str):
    status, _, raw = _fetch(f"http://export.arxiv.org/api/query?id_list={quote(arxiv_id)}")
    if status != 200 or not raw:
        return None, None, status
    try:
        root = ET.fromstring(_decode(raw))
    except ET.ParseError:
        return None, None, status
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find(".//a:entry", ns)
    if entry is None:
        return None, None, status
    summ = entry.find("a:summary", ns)
    titl = entry.find("a:title", ns)
    abstract = re.sub(r"\s+", " ", summ.text).strip() if summ is not None and summ.text else None
    title = re.sub(r"\s+", " ", titl.text).strip() if titl is not None and titl.text else None
    return abstract, title, status


def fetch_crossref(doi: str):
    status, _, raw = _fetch(f"https://api.crossref.org/works/{quote(doi)}", accept="application/json")
    if status != 200 or not raw:
        return None, None, status
    try:
        msg = json.loads(_decode(raw)).get("message", {})
    except json.JSONDecodeError:
        return None, None, status
    title = " ".join(msg.get("title") or []) or None
    abstract = msg.get("abstract")
    abstract = _clean_html(abstract) if abstract else None
    return (abstract or None), title, status


def fetch_semantic_scholar(id_kind: str, id_value: str):
    key = quote(f"{id_kind}:{id_value}", safe=":")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{key}?fields=abstract,title"
    status, _, raw = _fetch(url, accept="application/json")
    if status != 200 or not raw:
        return None, None, status
    try:
        d = json.loads(_decode(raw))
    except json.JSONDecodeError:
        return None, None, status
    return (d.get("abstract") or None), (d.get("title") or None), status


def fetch_openalex(doi: str):
    """OpenAlex reconstructs abstracts from an inverted index for a very broad
    range of works (including ACM/IEEE papers whose publishers block scraping
    and whose abstracts are absent from Crossref). Returns (abstract, title,
    status)."""
    status, _, raw = _fetch(f"https://api.openalex.org/works/doi:{quote(doi)}",
                            accept="application/json")
    if status != 200 or not raw:
        return None, None, status
    try:
        d = json.loads(_decode(raw))
    except json.JSONDecodeError:
        return None, None, status
    inv = d.get("abstract_inverted_index")
    title = d.get("title")
    if not inv:
        return None, title, status
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    abstract = " ".join(pos[i] for i in sorted(pos))
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return (abstract or None), title, status


def _title_overlap(a: str, b: str) -> float:
    """Jaccard overlap of word sets - a cheap guard against matching the wrong
    paper in a free-text title search."""
    wa = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if len(w) > 2}
    wb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if len(w) > 2}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# Publisher metadata placeholders that are not real titles, so a low overlap
# against them is not evidence of a wrong record.
_PLACEHOLDER_TITLES = {"oup accepted manuscript", "untitled", ""}


def _title_match_score(our: str, seen: str):
    """Overlap of our title with the title the API returned. Returns a float in
    [0,1], or None if it cannot be judged (missing title or a placeholder)."""
    if not our or not seen:
        return None
    s = re.sub(r"<[^>]+>", " ", seen).strip()
    if s.lower() in _PLACEHOLDER_TITLES:
        return None
    ta, tb = (
        {w for w in re.findall(r"[a-z0-9]+", our.lower()) if len(w) > 2},
        {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2},
    )
    if not ta or not tb:
        return None
    if ta <= tb or tb <= ta:  # one title is a truncation/superset of the other
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _consistent(our: str, seen: str, min_overlap: float = 0.3) -> bool:
    """Reject an abstract whose returned title clearly belongs to a different
    work (the wrong-DOI case). Lenient: only blocks a confident mismatch."""
    score = _title_match_score(our, seen)
    return score is None or score >= min_overlap


def fetch_semantic_scholar_by_title(title: str, min_overlap: float = 0.6):
    """Recover an abstract for a source that has no DOI/arXiv id by searching
    Semantic Scholar for its title. Only accept a hit whose title closely
    matches ours, so we never attach a different paper's abstract."""
    if not title or len(title) < 12:
        return None, None, None
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?query="
        + quote(title)
        + "&limit=3&fields=abstract,title"
    )
    status, _, raw = _fetch(url, accept="application/json")
    if status != 200 or not raw:
        return None, None, status
    try:
        hits = json.loads(_decode(raw)).get("data", []) or []
    except json.JSONDecodeError:
        return None, None, status
    # Choose the BEST-matching hit that has an abstract, not merely the first.
    best = max(
        ((h, _title_overlap(title, h.get("title", ""))) for h in hits if h.get("abstract")),
        key=lambda hc: hc[1], default=(None, 0.0),
    )
    if best[0] and best[1] >= min_overlap:
        return best[0]["abstract"], best[0].get("title"), status
    return None, None, status


def fetch_openalex_by_title(title: str, min_overlap: float = 0.6):
    """Recover an abstract via OpenAlex free-text search, with a match guard."""
    if not title or len(title) < 12:
        return None, None, None
    url = "https://api.openalex.org/works?search=" + quote(title) + "&per-page=3"
    status, _, raw = _fetch(url, accept="application/json")
    if status != 200 or not raw:
        return None, None, status
    try:
        results = json.loads(_decode(raw)).get("results", []) or []
    except json.JSONDecodeError:
        return None, None, status
    for w in results:
        inv = w.get("abstract_inverted_index")
        if inv and _title_overlap(title, w.get("title") or "") >= min_overlap:
            pos: dict[int, str] = {}
            for word, idxs in inv.items():
                for i in idxs:
                    pos[i] = word
            abstract = re.sub(r"\s+", " ", " ".join(pos[i] for i in sorted(pos))).strip()
            if abstract:
                return abstract, w.get("title"), status
    return None, None, status


def fetch_arxiv_by_title(title: str, min_overlap: float = 0.6):
    """Recover an abstract via an arXiv title search, with a match guard."""
    if not title or len(title) < 12:
        return None, None, None
    url = "http://export.arxiv.org/api/query?search_query=" + quote(f'ti:"{title}"') + "&max_results=3"
    status, _, raw = _fetch(url)
    if status != 200 or not raw:
        return None, None, status
    try:
        root = ET.fromstring(_decode(raw))
    except ET.ParseError:
        return None, None, status
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//a:entry", ns):
        t, s = entry.find("a:title", ns), entry.find("a:summary", ns)
        wt = re.sub(r"\s+", " ", t.text).strip() if t is not None and t.text else ""
        if s is not None and s.text and _title_overlap(title, wt) >= min_overlap:
            return re.sub(r"\s+", " ", s.text).strip(), wt, status
    return None, None, status


def _extract_pdf(raw: bytes, budget: int = WEB_CHAR_BUDGET, max_pages: int = 6):
    """Extract the first ~budget chars of an already-downloaded PDF."""
    if not raw:
        return None
    try:
        import io

        import pdfplumber  # type: ignore

        chunks: list[str] = []
        total = 0
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages[:max_pages]:
                t = page.extract_text() or ""
                chunks.append(t)
                total += len(t)
                if total >= budget * 2:
                    break
        text = re.sub(r"\s+", " ", " ".join(chunks)).strip()
        return text[:budget] or None
    except Exception:  # noqa: BLE001 - corrupt/encrypted PDF, missing parser
        return None


# --------------------------------------------------------------------------- #
# web-page extractor (descriptions for white papers, blogs, web resources)
# --------------------------------------------------------------------------- #
_META_RE = re.compile(
    r'<meta[^>]+(?:name|property)\s*=\s*["\'](?:description|og:description|twitter:description)["\'][^>]*>',
    re.I,
)
_CONTENT_RE = re.compile(r'content\s*=\s*["\'](.*?)["\']', re.I | re.S)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_SCRIPT_RE = re.compile(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", re.I | re.S)
_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.I | re.S)
# Pages that returned a JS-disabled / session / cookie stub rather than content.
_BOILERPLATE_RE = re.compile(
    r"enable javascript|tab window reload|refresh.{0,3}session|please enable|"
    r"javascript is (disabled|required)|browser.{0,15}(not )?support",
    re.I,
)


def _looks_boilerplate(text: str) -> bool:
    return bool(_BOILERPLATE_RE.search(text)) and len(text) < 400


def _meta_description(html: str) -> str:
    best = ""
    for m in _META_RE.finditer(html):
        c = _CONTENT_RE.search(m.group(0))
        if c:
            v = _clean_html(c.group(1))
            if len(v) > len(best):
                best = v
    return best


def _page_title(html: str) -> str:
    m = _TITLE_RE.search(html)
    return _clean_html(m.group(1)) if m else ""


def _lead_paragraphs(html: str, budget: int = WEB_CHAR_BUDGET) -> str:
    h = _SCRIPT_RE.sub(" ", html)
    out: list[str] = []
    for p in _P_RE.findall(h):
        t = _clean_html(p)
        if len(t) >= 40:
            out.append(t)
        if sum(len(x) for x in out) >= budget:
            break
    return " ".join(out)[:budget]


def fetch_webpage(url: str):
    # Read with a large cap so a PDF is not truncated; HTML pages stop at EOF
    # well before this, so there is no second request for the PDF path.
    status, ctype, raw = _fetch(url, ua=BROWSER_UA, max_bytes=PDF_MAX_BYTES)
    if status != 200 or not raw:
        return None, None, status
    if (ctype and "pdf" in ctype) or url.lower().split("?")[0].endswith(".pdf"):
        return (_extract_pdf(raw), "__PDF__", status)
    html = _decode(raw)
    title = _page_title(html)
    desc = _meta_description(html)
    text = desc
    if len(text) < 250:
        body = _lead_paragraphs(html)
        text = (text + " " + body).strip() if body else text
    text = text[: WEB_CHAR_BUDGET]
    if text and _looks_boilerplate(text):
        return None, title, status  # JS/session stub, not real content
    return (text or None), title, status


# --------------------------------------------------------------------------- #
# per-source orchestration
# --------------------------------------------------------------------------- #
def _override(source_id: str) -> str | None:
    p = CACHE_DIR / f"{source_id}.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def acquire_for_row(row: pd.Series) -> dict:
    sid = row["id"]
    out = {
        "id": sid, "number": row.get("number"),
        "text_strategy": _s(row.get("text_strategy")) or "manual",
        "source": "", "status": "empty", "http_status": "", "title_seen": "",
        "title_match": "", "text": "",
    }

    ov = _override(sid)
    if ov:
        out.update(source="override", status="ok", text=ov[:ABSTRACT_CHAR_BUDGET])
        return out

    strategy = out["text_strategy"]
    doi, arxiv, url = _s(row.get("doi")), _s(row.get("arxiv_id")), _s(row.get("url"))
    title = _s(row.get("title"))

    def _accept(text, source, st, seen) -> bool:
        """Accept an id-based hit only if the returned title is consistent with
        ours - this is the guard that catches a wrong/stale DOI attaching the
        wrong paper's abstract. Records the title-match score for auditing."""
        if not _consistent(title, seen):
            return False
        score = _title_match_score(title, seen)
        out.update(text=text[:ABSTRACT_CHAR_BUDGET], source=source, status="ok",
                   http_status=st, title_seen=seen or "",
                   title_match=("" if score is None else round(score, 2)))
        return True

    def _try_web(guard: bool = False) -> bool:
        if not url:
            return False
        txt, seen, st = fetch_webpage(url)
        out["http_status"] = st
        if txt:
            is_pdf = seen == "__PDF__"
            # For abstract sources the landing page should be the paper itself;
            # if its title disagrees with ours the URL/DOI points elsewhere, so
            # prefer the title search rather than attach a different work.
            if guard and not is_pdf and not _consistent(title, seen):
                return False
            score = None if is_pdf else _title_match_score(title, seen)
            out.update(text=txt[:ABSTRACT_CHAR_BUDGET], status="ok",
                       source=("pdf" if is_pdf else "web_page"),
                       title_seen="" if is_pdf else (seen or ""),
                       title_match=("" if score is None else round(score, 2)))
            return True
        if seen == "__PDF__":  # PDF found but could not be parsed
            out.update(source="pdf_no_parser", status="fallback")
        return False

    if strategy == "abstract":
        # By-id (arXiv -> Crossref -> Semantic Scholar -> OpenAlex), each
        # guarded against a wrong DOI; then the landing page; then a guarded
        # title search across Semantic Scholar, OpenAlex and arXiv.
        if arxiv:
            txt, seen, st = fetch_arxiv(arxiv)
            if txt and _accept(txt, "arxiv", st, seen):
                return out
        if doi:
            for fn, label in ((fetch_crossref, "crossref"),
                              (lambda d: fetch_semantic_scholar("DOI", d), "semantic_scholar"),
                              (fetch_openalex, "openalex")):
                txt, seen, st = fn(doi)
                if txt and _accept(txt, label, st, seen):
                    return out
        if arxiv:
            txt, seen, st = fetch_semantic_scholar("ARXIV", arxiv)
            if txt and _accept(txt, "semantic_scholar", st, seen):
                return out
        if _try_web(guard=True):
            return out
        for fn, label in ((fetch_semantic_scholar_by_title, "semantic_scholar_title"),
                          (fetch_openalex_by_title, "openalex_title"),
                          (fetch_arxiv_by_title, "arxiv_title")):
            txt, seen, st = fn(title)
            if txt:  # title-search fns already enforce the match guard internally
                score = _title_match_score(title, seen)
                out.update(text=txt[:ABSTRACT_CHAR_BUDGET], source=label, status="ok",
                           http_status=st, title_seen=seen or "",
                           title_match=("" if score is None else round(score, 2)))
                return out
    else:
        # summary / full / manual: the landing page is the description source.
        if _try_web():
            return out
        if doi:  # a few summary items carry a DOI (e.g. NIST CSWP)
            txt, seen, st = fetch_crossref(doi)
            if txt and _accept(txt, "crossref", st, seen):
                return out

    if out["status"] == "empty":
        out["status"] = "fallback"
    if out["status"] == "fallback" and not out["source"]:
        out["source"] = "title_fallback"  # explicit provenance: no abstract acquired
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", default="data/sources.csv")
    ap.add_argument("--out", default="data/source_text.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write-cache", action="store_true",
                    help="also write data/abstracts_cache/{id}.json for fetched payloads")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.sources)
    rows: list[dict] = [None] * len(df)  # type: ignore
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire_for_row, row): i for i, (_, row) in enumerate(df.iterrows())}
        for fut in futures:
            rows[futures[fut]] = fut.result()

    for r in rows:
        r["n_chars"] = len(r["text"])
    out_df = pd.DataFrame(rows)[
        ["id", "number", "text_strategy", "source", "status", "http_status",
         "n_chars", "title_match", "title_seen", "text"]
    ]
    out_df.to_csv(ROOT / args.out, index=False)

    if args.write_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r["status"] == "ok" and r["source"] != "override":
                (CACHE_DIR / f"{r['id']}.json").write_text(
                    json.dumps({k: r[k] for k in ("id", "source", "title_seen", "text")}, indent=2),
                    encoding="utf-8",
                )

    n = len(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"acquired text for {ok}/{n} sources -> {args.out}")
    print("\nby source:")
    print(out_df["source"].value_counts().to_string())
    print("\nby strategy x status:")
    print(pd.crosstab(out_df["text_strategy"], out_df["status"]).to_string())


if __name__ == "__main__":
    main()
