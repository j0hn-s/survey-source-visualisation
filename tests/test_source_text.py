"""Coverage and integrity tests for the acquired text (data/source_text.csv).

These let you confirm that *every* source had its abstract (or type-appropriate
equivalent) considered, see where each came from, and catch the case where an
abstract was attached to the wrong paper.

`source_text.csv` is a committed, reproducible artefact (regenerate with
`python -m src.acquire_text`). If it is absent the tests skip rather than fail,
because regeneration needs network access (an integration concern).
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from src import build_figures as bf

# The per-type text model: which acquisition strategy each publication type
# uses, and the assumption behind it. This table IS the modelling decision, so
# the test pins it down.
TYPE_TO_STRATEGY = {
    "academic_paper": "abstract",
    "preprint": "abstract",
    "survey_review": "abstract",
    "regulatory_guidance": "summary",
    "technical_whitepaper": "summary",
    "standards_specification": "summary",
    "industry_blog": "full",
    "online_resource": "manual",
    "software_repository": "manual",
}


@pytest.fixture(scope="module")
def texts():
    if not bf.SOURCE_TEXT.exists():
        pytest.skip("data/source_text.csv not present; run `python -m src.acquire_text`")
    return pd.read_csv(bf.SOURCE_TEXT)


@pytest.fixture(scope="module")
def sources():
    return pd.read_csv(bf.SOURCES)


def test_every_source_has_exactly_one_text_row(texts, sources):
    assert len(texts) == len(sources)
    assert set(texts["id"]) == set(sources["id"])  # nothing skipped, nothing extra
    assert texts["id"].is_unique


def test_every_source_has_a_strategy(texts):
    assert texts["text_strategy"].notna().all()
    assert (texts["text_strategy"].astype(str).str.len() > 0).all()
    assert set(texts["text_strategy"]) <= {"abstract", "summary", "full", "manual"}


def test_strategy_matches_publication_type(sources):
    """The type-aware text model is applied to every source."""
    seed = sources["publication_type_seed"].fillna("")
    for ptype, expected in TYPE_TO_STRATEGY.items():
        got = sources.loc[seed == ptype, "text_strategy"].unique().tolist()
        assert got == [expected], f"{ptype}: expected {expected}, got {got}"


def test_status_is_ok_or_fallback_and_ok_has_text(texts):
    assert set(texts["status"]) <= {"ok", "fallback"}
    ok = texts[texts["status"] == "ok"]
    assert (ok["text"].fillna("").astype(str).str.len() > 0).all()
    # fallbacks are recorded, not silently dropped: they keep a strategy + source label
    fb = texts[texts["status"] == "fallback"]
    assert fb["source"].notna().all()


def test_coverage_is_reported_and_reasonable(texts):
    n = len(texts)
    ok = (texts["status"] == "ok").sum()
    pct = 100 * ok / n
    print(f"\nabstract/description coverage: {ok}/{n} ({pct:.1f}%)")
    print(texts["source"].value_counts().to_string())
    # Floor set below the achieved coverage (~88%) with enough headroom to
    # absorb transient API flakiness, but tight enough that losing a whole
    # acquisition path (e.g. all OpenAlex hits) trips it.
    assert pct >= 70.0


def test_api_abstracts_match_their_source(texts, sources):
    """Guard against attaching an abstract from the wrong paper: for sources
    whose abstract came from a bibliographic API (which also returns the
    record's title), the returned title should resemble our title. Gross
    mismatches are listed so the underlying reference (likely a wrong DOI) can
    be checked."""
    title = dict(zip(sources["id"], sources["title"].fillna("")))
    # Publisher metadata placeholders that are not real titles.
    placeholders = {"oup accepted manuscript", "untitled"}

    def toks(s):
        s = re.sub(r"<[^>]+>", " ", str(s))  # strip HTML (e.g. "<i>L</i>-diversity")
        return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}

    mismatches = []
    for _, r in texts.iterrows():
        # Any source that returned a title (API or guarded web page) can be
        # cross-checked against our title. Rows with no returned title
        # (e.g. title_fallback) have NaN here and are skipped.
        ours = title.get(r["id"], "")
        seen = str(r["title_seen"]) if pd.notna(r["title_seen"]) else ""
        if not ours.strip() or not seen.strip() or seen.strip().lower() in placeholders:
            continue
        ta, tb = toks(ours), toks(seen)
        if not ta or not tb or ta <= tb or tb <= ta:
            continue  # identical, or one title is a truncation/superset of the other
        overlap = len(ta & tb) / len(ta | tb)
        if overlap < 0.3:
            mismatches.append((r["id"], round(overlap, 2), ours[:50], seen[:50]))

    if mismatches:
        print("\nLOW title overlap (verify these references' DOIs):")
        for m in mismatches:
            print("  ", m)
    # A handful is tolerable (hyphenation/HTML artefacts); a flood means the
    # extraction is grabbing wrong records.
    assert len(mismatches) <= 4
