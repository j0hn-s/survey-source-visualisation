"""Integration tests for src.build_figures - the end-to-end figure build.

This runs the real headless build once (session-scoped) against the real
`data/sources.csv` and then asserts the invariants a reader would want to
check before trusting the figures:

  * the graph has exactly one node per source - nothing dropped or duplicated;
  * every source is assigned to a cluster, and cluster sizes sum to N;
  * the map has the edges it should (semantic + family, no topic edges);
  * the clustering finds real structure (modularity well above zero);
  * publication type is fully populated (no "Unclassified" left);
  * publication years are sane and the count of undated sources is reported;
  * all four figure files are written and non-empty;
  * the on-disk metrics JSON is internally consistent with the build.

These are deliberately invariant checks, not golden-image comparisons: they
catch a broken pipeline (a stage silently dropping rows, a column rename, an
empty figure) without being brittle to a one-pixel rendering change.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src import build_figures as bf


@pytest.fixture(scope="session")
def built():
    """Run the build once; expose (metrics, sources_df)."""
    metrics = bf.build(quiet=True)
    sources = pd.read_csv(bf.SOURCES)
    return metrics, sources


def test_one_node_per_source(built):
    metrics, sources = built
    assert metrics["n_sources"] == len(sources)
    assert metrics["n_nodes"] == len(sources)


def test_every_source_is_clustered(built):
    metrics, sources = built
    clusters = pd.read_csv(bf.CLUSTERS_CSV)
    assert len(clusters) == len(sources)
    assert clusters["cluster"].notna().all()
    assert (clusters["cluster"].astype(str).str.len() > 0).all()
    # ids line up one-to-one with the source inventory.
    assert set(clusters["id"]) == set(sources["id"])


def test_cluster_sizes_sum_to_n(built):
    metrics, sources = built
    total = sum(metrics["cluster_sizes"].values()) + metrics["n_other"]
    assert total == len(sources)


def test_map_has_semantic_and_family_edges_only(built):
    metrics, _ = built
    by_type = metrics["edges_by_type"]
    assert by_type.get("semantic", 0) > 0
    assert by_type.get("family", 0) > 0
    # The science-mapping build colours by cluster and adds no topic edges.
    assert "topic" not in by_type


def test_clustering_finds_real_structure(built):
    metrics, _ = built
    assert metrics["n_clusters"] >= 3
    assert metrics["modularity"] > 0.3


def test_text_coverage_reported(built):
    metrics, sources = built
    tc = metrics["text_coverage"]
    assert tc["n_acquired"] + tc["n_fallback"] == len(sources)
    # The clusters are meant to come from abstracts/descriptions, so the bulk
    # of sources must carry acquired text rather than a title fallback.
    assert tc["pct_acquired"] >= 70.0


def test_cluster_labels_are_present_and_distinct(built):
    metrics, _ = built
    labels = metrics["cluster_labels"]
    assert set(labels) == set(metrics["cluster_sizes"])
    assert all(v.strip() for v in labels.values())
    assert len(set(labels.values())) == len(labels)  # no two clusters share a label


def test_publication_type_fully_populated(built):
    metrics, sources = built
    dist = metrics["type_distribution"]
    assert sum(dist.values()) == len(sources)
    assert "(none)" not in dist  # every source seeds to a known type


def test_years_are_sane(built):
    metrics, sources = built
    assert 1980 <= metrics["year_min"] <= metrics["year_max"] <= 2027
    expected_missing = int(pd.to_numeric(sources["year"], errors="coerce").isna().sum())
    assert metrics["n_year_missing"] == expected_missing


def test_all_figures_written_and_nonempty(built):
    metrics, _ = built
    expected = {
        "semantic_map_main.png", "timeline.png",
        "type_topic_heatmap.png", "pet_family_breakdown.png",
    }
    assert expected.issubset(set(metrics["figures"]))
    for name in expected:
        path = bf.FIGURES / name
        assert path.exists() and path.stat().st_size > 0


def test_metrics_file_matches_return_value(built):
    metrics, _ = built
    on_disk = json.loads(bf.METRICS.read_text(encoding="utf-8"))
    assert on_disk["n_nodes"] == metrics["n_nodes"]
    assert on_disk["n_clusters"] == metrics["n_clusters"]
    assert on_disk["modularity"] == metrics["modularity"]
