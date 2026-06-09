"""Offline tests for src.acquire_text - the Stage 2 text acquisition.

No network: the HTML/JSON extractors are tested on fixed inputs, and the
per-source routing is tested by monkeypatching the fetchers. The point is to
verify *how* text is acquired and that the type-aware escalation picks the
right source - the assumptions the semantic clusters rest on.
"""
from __future__ import annotations

import json

import pandas as pd

from src import acquire_text as at


# --- pure extractors ------------------------------------------------------- #
def test_clean_html_strips_tags_and_entities():
    assert at._clean_html("<p>Hello&amp;  <b>world</b></p>") == "Hello& world"


def test_meta_description_prefers_longest():
    html = (
        '<meta name="description" content="short">'
        '<meta property="og:description" content="a much longer description here">'
    )
    assert at._meta_description(html) == "a much longer description here"


def test_page_title():
    assert at._page_title("<title>Privacy Sandbox &mdash; overview</title>") == "Privacy Sandbox — overview"


def test_lead_paragraphs_skips_short_and_concatenates():
    html = "<p>tiny</p><p>" + "x" * 60 + "</p><p>" + "y" * 60 + "</p>"
    out = at._lead_paragraphs(html, budget=1000)
    assert "x" * 60 in out and "y" * 60 in out and "tiny" not in out


def test_openalex_reconstructs_abstract_from_inverted_index(monkeypatch):
    inverted = {"Privacy": [0], "enhancing": [1], "technologies": [2], "matter": [3]}
    body = json.dumps({"title": "T", "abstract_inverted_index": inverted}).encode()
    monkeypatch.setattr(at, "_fetch", lambda *a, **k: (200, "application/json", body))
    abstract, title, status = at.fetch_openalex("10.1/x")
    assert abstract == "Privacy enhancing technologies matter"
    assert status == 200


def test_title_overlap_guard():
    assert at._title_overlap("Deep learning with differential privacy",
                             "Deep Learning with Differential Privacy") == 1.0
    assert at._title_overlap("federated learning", "homomorphic encryption") == 0.0


def test_s2_title_search_rejects_low_overlap_match(monkeypatch):
    # Returns a hit whose title does not match -> must be rejected.
    data = {"data": [{"title": "An unrelated paper about cats", "abstract": "meow"}]}
    monkeypatch.setattr(at, "_fetch", lambda *a, **k: (200, "application/json", json.dumps(data).encode()))
    abstract, _, _ = at.fetch_semantic_scholar_by_title("Secure multi-party computation protocols")
    assert abstract is None


def test_s2_title_search_accepts_matching_title(monkeypatch):
    data = {"data": [{"title": "Secure multi-party computation protocols", "abstract": "the body"}]}
    monkeypatch.setattr(at, "_fetch", lambda *a, **k: (200, "application/json", json.dumps(data).encode()))
    abstract, _, _ = at.fetch_semantic_scholar_by_title("Secure multi-party computation protocols")
    assert abstract == "the body"


# --- per-source routing (the type-aware escalation) ------------------------ #
def _row(**kw):
    # title defaults to "" so pure-routing tests bypass the title-match guard;
    # guard tests pass an explicit title.
    base = dict(id="ref-x", number=1, text_strategy="abstract", doi="", arxiv_id="", url="", title="")
    base.update(kw)
    return pd.Series(base)


def _stub(monkeypatch, **overrides):
    """Make every fetcher return 'nothing' unless overridden with (txt, seen, status)."""
    none = (None, None, None)
    for name in ("fetch_arxiv", "fetch_crossref", "fetch_semantic_scholar",
                 "fetch_openalex", "fetch_semantic_scholar_by_title",
                 "fetch_openalex_by_title", "fetch_arxiv_by_title"):
        monkeypatch.setattr(at, name, lambda *a, _r=overrides.get(name, none), **k: _r)
    monkeypatch.setattr(at, "fetch_webpage", lambda *a, _r=overrides.get("fetch_webpage", none), **k: _r)


def test_abstract_strategy_prefers_arxiv(monkeypatch):
    _stub(monkeypatch, fetch_arxiv=("arxiv abstract", "seen", 200),
          fetch_crossref=("cr", "x", 200))
    out = at.acquire_for_row(_row(arxiv_id="2501.00001", doi="10.1/x"))
    assert out["source"] == "arxiv" and out["status"] == "ok"


def test_abstract_strategy_escalates_to_openalex(monkeypatch):
    # arXiv absent; crossref + s2 empty; openalex has it.
    _stub(monkeypatch, fetch_openalex=("oa abstract", "seen", 200))
    out = at.acquire_for_row(_row(doi="10.1/x"))
    assert out["source"] == "openalex" and out["status"] == "ok"


def test_summary_strategy_uses_webpage(monkeypatch):
    _stub(monkeypatch, fetch_webpage=("a description", "Page Title", 200))
    out = at.acquire_for_row(_row(text_strategy="summary", url="https://example.org"))
    assert out["source"] == "web_page" and out["status"] == "ok"


def test_crossref_preferred_over_openalex(monkeypatch):
    _stub(monkeypatch, fetch_crossref=("cr abstract", "seen", 200),
          fetch_openalex=("oa abstract", "seen", 200))
    out = at.acquire_for_row(_row(doi="10.1/x"))
    assert out["source"] == "crossref"


def test_falls_back_when_nothing_found(monkeypatch):
    _stub(monkeypatch)  # everything returns nothing
    out = at.acquire_for_row(_row(text_strategy="abstract", doi="10.1/x", url="https://x"))
    assert out["status"] == "fallback" and out["text"] == ""


def test_wrong_doi_abstract_is_rejected_by_title_guard(monkeypatch):
    # The DOI resolves to a different paper (mismatched title). The guard must
    # refuse to attach it; with no other source it falls back. This is the
    # #152 wrong-DOI class.
    _stub(monkeypatch, fetch_crossref=("abstract of a totally different paper",
                                       "An unrelated paper about feline behaviour", 200))
    out = at.acquire_for_row(_row(doi="10.1/x",
                                  title="Secure multi-party computation protocols for finance"))
    assert out["source"] != "crossref"
    assert out["status"] == "fallback"


def test_matching_title_is_accepted_and_records_title_match(monkeypatch):
    _stub(monkeypatch, fetch_crossref=("the real abstract",
                                       "Secure multi-party computation protocols for finance", 200))
    out = at.acquire_for_row(_row(doi="10.1/x",
                                  title="Secure multi-party computation protocols for finance"))
    assert out["source"] == "crossref" and out["status"] == "ok"
    assert out["title_match"] == 1.0


def test_guarded_web_falls_through_to_title_search(monkeypatch):
    # Landing page is about a different work (wrong URL/DOI); the guarded web
    # hit must be refused so the title search can recover the right paper.
    _stub(monkeypatch,
          fetch_webpage=("description of the wrong work", "A different paper entirely", 200),
          fetch_openalex_by_title=("correct abstract via title search",
                                   "Our Real Paper Title", 200))
    out = at.acquire_for_row(_row(url="https://x", title="Our Real Paper Title"))
    assert out["source"] == "openalex_title" and out["status"] == "ok"


def test_override_always_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(at, "CACHE_DIR", tmp_path)
    (tmp_path / "ref-x.txt").write_text("human supplied excerpt", encoding="utf-8")
    out = at.acquire_for_row(_row(arxiv_id="2501.00001"))
    assert out["source"] == "override" and out["text"] == "human supplied excerpt"
