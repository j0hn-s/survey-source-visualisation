"""Tests for src.graph.

The three edge rules (topic / family / semantic) are independent and each
should be testable in isolation. We verify edge presence, edge typing, and
that the k-NN cap on semantic edges is honoured.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from src import graph as graph_mod


def _edge_types(g):
    return Counter(d["etype"] for _, _, d in g.edges(data=True))


def test_topic_edges_link_shared_primary_topic(sample_metadata):
    # Disable semantic edges by passing an all-zero similarity matrix.
    n = len(sample_metadata)
    g = graph_mod.build_graph(sample_metadata, similarity=np.zeros((n, n)))

    types = _edge_types(g)
    # refA and refB do not share a primary topic, so no topic edge directly
    # between them. refC and refD do not share a primary topic either.
    # In the fixture every source has a distinct primary topic, so we expect
    # zero topic edges.
    assert types.get("topic", 0) == 0


def test_topic_edges_are_created_for_shared_topic(sample_metadata):
    df = sample_metadata.copy()
    # Force refA and refB to share a topic.
    df.loc[df["id"] == "refB", "primary_topic"] = "mechanism"
    n = len(df)
    g = graph_mod.build_graph(df, similarity=np.zeros((n, n)))
    topic_pairs = [
        tuple(sorted([u, v]))
        for u, v, d in g.edges(data=True)
        if d["etype"] == "topic"
    ]
    assert ("refA", "refB") in topic_pairs


def test_family_edges_are_multi_label(sample_metadata):
    n = len(sample_metadata)
    g = graph_mod.build_graph(sample_metadata, similarity=np.zeros((n, n)))
    fam_pairs = [
        tuple(sorted([u, v]))
        for u, v, d in g.edges(data=True)
        if d["etype"] == "family"
    ]
    # refA, refB, and refC all carry differential_privacy.
    assert ("refA", "refB") in fam_pairs
    assert ("refA", "refC") in fam_pairs
    assert ("refB", "refC") in fam_pairs
    # refD carries no family -> no family edges involving refD.
    assert all("refD" not in pair for pair in fam_pairs)


def test_semantic_edges_respect_threshold_and_k(sample_metadata):
    n = len(sample_metadata)
    # Strong similarity refA-refC; everything else just under threshold.
    sim = np.full((n, n), 0.10)
    np.fill_diagonal(sim, 1.0)
    ids = sample_metadata["id"].tolist()
    a = ids.index("refA")
    c = ids.index("refC")
    sim[a, c] = sim[c, a] = 0.9

    g = graph_mod.build_graph(
        sample_metadata,
        similarity=sim,
        k_semantic=2,
        semantic_threshold=0.25,
    )
    semantic_pairs = [
        tuple(sorted([u, v]))
        for u, v, d in g.edges(data=True)
        if d["etype"] == "semantic"
    ]
    assert ("refA", "refC") in semantic_pairs
    # No other pair clears the 0.25 threshold.
    assert len(semantic_pairs) == 1


def test_filter_edges_keeps_only_requested_types(sample_metadata):
    n = len(sample_metadata)
    g = graph_mod.build_graph(sample_metadata, similarity=np.zeros((n, n)))
    filtered = graph_mod.filter_edges(g, {"family"})
    assert all(d["etype"] == "family" for _, _, d in filtered.edges(data=True))
    assert filtered.number_of_nodes() == g.number_of_nodes()


def test_safe_year_handles_blank_and_string(sample_metadata):
    """Mirror what `pd.read_csv(...).fillna('')` produces - an object column
    where some entries are blank strings and others are numeric strings.
    """
    df = sample_metadata.copy()
    df["year"] = df["year"].astype(object)
    df.loc[df["id"] == "refD", "year"] = ""
    df.loc[df["id"] == "refA", "year"] = "2020"
    g = graph_mod.build_graph(df, similarity=np.zeros((len(df), len(df))))
    assert g.nodes["refA"]["year"] == 2020
    assert g.nodes["refD"]["year"] is None
