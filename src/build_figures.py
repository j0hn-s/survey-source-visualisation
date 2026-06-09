"""End-to-end, headless rebuild of the survey-source figures.

This script reproduces notebook stages 3-7 (normalise -> TF-IDF -> topic
scoring -> graph -> render) without a running kernel, so the figures can be
regenerated deterministically from `data/sources.csv` alone:

    python -m src.build_figures            # rebuild every figure
    python -m src.build_figures --quiet    # same, without the console report

Design decisions worth knowing before you change anything:

  * Content text per source is the TITLE. The repository ships no abstract
    cache (acquiring third-party abstracts is out of scope), and the
    bibliographic venue string is deliberately excluded: it carries
    journal/publisher names ("...Law Review", "Foundations and Trends",
    "Proceedings of...") that collide with topic keywords and couple
    unrelated papers that merely share a publisher. For the 77 sources
    without a quoted title (reports, standards, web pages) the content text
    is the leading descriptive clause of `venue` (the report title), with
    the publisher/"Available at:" tail dropped. See `_content_text`.

  * Two *different* text forms are fed to the two scorers, deliberately:
      - the SEMANTIC TF-IDF (edge construction) uses the NORMALISED text
        (lower-cased, lemmatised, PET phrases joined) so paraphrases align;
      - the TOPIC scorer uses the RAW (un-normalised) text, because it
        compares against raw keyword phrases ("differential privacy",
        "garbled circuit"). Normalising would fuse those into single tokens
        and destroy the bigram match against the topic pseudo-documents.

  * Column resolution is curated-over-seed: a human value in
    `publication_type` / `pet_family` / `primary_topic` always wins; the
    mechanical *_seed (and the topic suggestion) is the fallback. Today no
    curated overrides exist, so the figures are fully mechanical and
    reproducible; adding a review later changes nothing about how this runs.

A machine-readable summary is written to `data/figure_metrics.json` so the
tests in `tests/test_figures.py` can assert against the exact same numbers a
reader sees in the console report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.features import compute_tfidf, pairwise_cosine
from src.graph import build_graph
from src.semantic_clusters import cluster_corpus
from src.visualise import render_graph, render_supporting

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources.csv"
SOURCE_TEXT = ROOT / "data" / "source_text.csv"
VOCAB = ROOT / "data" / "topic_vocabulary.yaml"
FIGURES = ROOT / "figures"
CLUSTERS_CSV = ROOT / "data" / "semantic_clusters.csv"
METRICS = ROOT / "data" / "figure_metrics.json"

# All knobs live here, not buried in calls, because they are what a reviewer
# most wants to reason about.
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9
TFIDF_NGRAM = (1, 2)
MODEL_CHAR_BUDGET = 1800  # cap per source so a long abstract cannot dominate
SEED = 42

# Science-mapping (semantic clustering) parameters. The figure draws semantic
# edges from the SAME k-NN graph the clustering uses, so edges and clusters
# tell one consistent story. See src/semantic_clusters.py for the method.
# Tuned for the abstract-based text (richer than titles): a higher similarity
# threshold keeps the map from collapsing into a hairball.
CLUSTER_K = 12            # k nearest neighbours per source (VOSviewer-style cap)
CLUSTER_THRESHOLD = 0.10  # minimum cosine similarity for an edge
CLUSTER_RESOLUTION = 0.6  # Louvain resolution (lower -> fewer, larger clusters)
CLUSTER_MIN_SIZE = 4      # clusters smaller than this fold into "other"
CLUSTER_TOPN = 3          # characteristic terms per cluster label


def _lead_clause(venue: str) -> str:
    """The leading descriptive clause of a bibliographic venue string.

    For the 77 sources with no quoted title (reports, standards, web pages)
    the content-bearing text is the first sentence of `venue` - the report
    title - followed by a publisher/location/"Available at:" tail that is
    pure metadata. We keep the lead and drop the tail at the first sentence
    break so the publisher name does not pollute the content signal.
    """
    venue = venue.strip()
    cut = venue.find(". ")
    return venue[:cut] if cut > 0 else venue


def _content_text(row: pd.Series) -> str:
    """Content text for scoring: the title, or the report's leading clause.

    We deliberately exclude the bibliographic venue for titled sources. The
    venue carries journal/publisher names ("...Law Review", "Foundations and
    Trends", "Proceedings of...") that collide with topic keywords and create
    spurious semantic similarity between unrelated papers that merely share a
    publisher. The title is the substantive, content-bearing text.
    """
    title = (str(row["title"]) if pd.notna(row["title"]) else "").strip()
    if title:
        return title
    venue = (str(row["venue"]) if pd.notna(row["venue"]) else "").strip()
    return _lead_clause(venue)


def _normalise(texts: list[str]) -> list[str]:
    """Normalise via NLTK if available; otherwise degrade to lower-case.

    The notebook wraps the same call in try/except so the pipeline still runs
    on a machine without the NLTK corpora downloaded. We mirror that exactly:
    the fallback is weaker (no lemmatisation/stop-word removal) but keeps the
    build reproducible offline.
    """
    try:
        from src.preprocess import normalise_series

        return list(normalise_series(texts, VOCAB))
    except Exception as exc:  # noqa: BLE001 - any NLTK/download failure
        print(f"  [normalise] NLTK unavailable ({exc!s}); using lower-case fallback")
        return [t.lower() for t in texts]


def _resolve(curated, seed) -> str:
    c = "" if curated is None or (isinstance(curated, float) and pd.isna(curated)) else str(curated).strip()
    if c:
        return c
    s = "" if seed is None or (isinstance(seed, float) and pd.isna(seed)) else str(seed).strip()
    return s


def _load_source_text() -> dict:
    """id -> (acquired_text, status) from data/source_text.csv (Stage 2)."""
    if not SOURCE_TEXT.exists():
        return {}
    t = pd.read_csv(SOURCE_TEXT)
    return {
        r["id"]: ((str(r["text"]) if pd.notna(r["text"]) else "").strip(),
                  (str(r["status"]) if pd.notna(r["status"]) else "fallback"))
        for _, r in t.iterrows()
    }


def _model_text(row: pd.Series, acquired: dict) -> str:
    """The text fed to the semantic model: title joined to the acquired
    abstract / description, capped for length parity. When acquisition failed
    we fall back to the title (or, for untitled reports, the venue lead) so a
    source is never empty - the fallback is recorded in source_text.csv."""
    title = (str(row["title"]) if pd.notna(row["title"]) else "").strip()
    body, status = acquired.get(row["id"], ("", "fallback"))
    if status == "ok" and body:
        base = (title + ". " + body) if title else body
    else:
        base = _content_text(row)
    return base[:MODEL_CHAR_BUDGET]


def build(quiet: bool = False) -> dict:
    df = pd.read_csv(SOURCES)
    df = df.reset_index(drop=True)
    ids = df["id"].tolist()

    # --- stage 2 output: type-appropriate text per source (abstract for
    # academic/preprint/survey; description for white papers, blogs, web
    # resources; title fallback when acquisition failed). See acquire_text.py. ---
    acquired = _load_source_text()
    raw_text = [_model_text(r, acquired) for _, r in df.iterrows()]

    # --- stage 3+4: normalise -> semantic TF-IDF -> cosine similarity ---
    norm_text = _normalise(raw_text)
    feats = compute_tfidf(
        ids, norm_text, min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, ngram_range=TFIDF_NGRAM
    )
    sim = pairwise_cosine(feats.tfidf_matrix)

    # --- stage 5: emergent semantic clusters from the abstracts (figure colour) ---
    clusters = cluster_corpus(
        ids, raw_text, sim,
        k=CLUSTER_K, threshold=CLUSTER_THRESHOLD, resolution=CLUSTER_RESOLUTION,
        seed=SEED, min_size=CLUSTER_MIN_SIZE, top_n=CLUSTER_TOPN,
    )

    # --- resolve curated-over-seed into the metadata the graph consumes ---
    merged = pd.DataFrame(
        {
            "id": df["id"],
            "number": df["number"],
            "year": df["year"],
            "title": df["title"].fillna(""),
            "authors": df["authors"].fillna(""),
            "venue": df["venue"].fillna(""),
        }
    )
    merged["publication_type"] = [
        _resolve(c, s) for c, s in zip(df["publication_type"], df["publication_type_seed"])
    ]
    merged["cluster"] = [clusters.node2cluster.get(i, "other") for i in df["id"]]
    merged["pet_family"] = [
        _resolve(c, s) for c, s in zip(df["pet_family"], df["pet_family_seed"])
    ]
    # primary_topic is deliberately left out of the figure metadata so that no
    # topic edges form; the layout is driven by semantic + family edges only.

    # Persist the cluster assignment for inspection.
    cl_display = clusters.display()
    pd.DataFrame({
        "id": merged["id"], "number": merged["number"],
        "cluster": merged["cluster"],
        "cluster_label": [clusters.labels.get(c, "") for c in merged["cluster"]],
        "title": merged["title"],
    }).to_csv(CLUSTERS_CSV, index=False)

    # --- stage 6: graph (semantic + family edges; coloured by cluster) ---
    g = build_graph(merged, sim, k_semantic=CLUSTER_K, semantic_threshold=CLUSTER_THRESHOLD)

    # --- stage 7: render the two headline figures + supporting ---
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_graph(
        g, out_path=FIGURES / "semantic_map_main.png",
        title="Sources considered: semantic map (clusters from abstracts)",
        show_edges=("semantic",),
        color_attr="cluster", palette=clusters.palette, display=cl_display,
        legend_title="Semantic cluster (characteristic terms)",
        # Pure-semantic layout: position is driven only by abstract similarity,
        # so clusters separate cleanly. (Family ties would otherwise pull
        # same-PET nodes together across clusters and muddy the centre.)
        layout_weights={"semantic": 1.0, "family": 0.0, "topic": 0.0},
        edge_alpha=0.10,
        # Draw only the stronger links and label each cluster at its centroid,
        # so the map reads cleanly rather than as a hairball.
        semantic_draw_min=0.22,
        annotate_clusters=True,
    )
    render_supporting(
        merged, FIGURES,
        group_col="cluster", palette=clusters.palette, display=cl_display,
        group_title="Semantic cluster",
    )

    # --- metrics for inspection + tests ---
    edge_counts: dict[str, int] = {}
    for _, _, d in g.edges(data=True):
        et = d.get("etype", "?")
        edge_counts[et] = edge_counts.get(et, 0) + 1
    years = pd.to_numeric(merged["year"], errors="coerce").dropna().astype(int)
    n_ok = sum(1 for v in acquired.values() if v[1] == "ok")
    metrics = {
        "n_sources": int(len(merged)),
        "n_nodes": int(g.number_of_nodes()),
        "n_edges_total": int(g.number_of_edges()),
        "edges_by_type": edge_counts,
        "text_coverage": {
            "n_acquired": int(n_ok),
            "n_fallback": int(len(merged) - n_ok),
            "pct_acquired": round(100 * n_ok / max(len(merged), 1), 1),
        },
        "n_clusters": len(clusters.cluster_keys),
        "modularity": round(clusters.modularity, 4),
        "cluster_sizes": {k: clusters.sizes[k] for k in clusters.cluster_keys},
        "n_other": clusters.sizes.get("other", 0),
        "cluster_labels": {k: clusters.labels.get(k, "") for k in clusters.cluster_keys},
        "type_distribution": merged["publication_type"].fillna("").replace("", "(none)").value_counts().to_dict(),
        "year_min": int(years.min()) if len(years) else None,
        "year_max": int(years.max()) if len(years) else None,
        "n_year_missing": int(pd.to_numeric(merged["year"], errors="coerce").isna().sum()),
        "tfidf_features": int(len(feats.tfidf_vocab)),
        "params": {
            "tfidf_min_df": TFIDF_MIN_DF, "tfidf_max_df": TFIDF_MAX_DF,
            "tfidf_ngram": list(TFIDF_NGRAM), "cluster_k": CLUSTER_K,
            "cluster_threshold": CLUSTER_THRESHOLD, "cluster_resolution": CLUSTER_RESOLUTION,
            "cluster_min_size": CLUSTER_MIN_SIZE, "seed": SEED,
        },
        "figures": sorted(p.name for p in FIGURES.glob("*.png")),
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not quiet:
        _report(metrics)
    return metrics


def _report(m: dict) -> None:
    print("\n=== figure build summary ===")
    print(f"sources={m['n_sources']}  nodes={m['n_nodes']}  edges={m['n_edges_total']}  "
          f"(by type: {m['edges_by_type']})")
    tc = m["text_coverage"]
    print(f"text: {tc['n_acquired']}/{m['n_sources']} acquired abstracts/descriptions "
          f"({tc['pct_acquired']}%), {tc['n_fallback']} title fallback")
    print(f"tfidf features={m['tfidf_features']}  clusters={m['n_clusters']} "
          f"(modularity {m['modularity']}, other={m['n_other']})  "
          f"years={m['year_min']}-{m['year_max']} (missing {m['n_year_missing']})")
    print("\nsemantic clusters (characteristic terms):")
    for k, lbl in m["cluster_labels"].items():
        print(f"  {k:<4} n={m['cluster_sizes'][k]:<4} {lbl}")
    print("\nsource type:")
    for k, v in m["type_distribution"].items():
        print(f"  {k:<24} {v}")
    print(f"\nwrote figures: {', '.join(m['figures'])}")
    print(f"metrics -> {METRICS.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="suppress the console report")
    args = ap.parse_args()
    build(quiet=args.quiet)


if __name__ == "__main__":
    main()
