"""Tests for src.semantic_clusters - the science-mapping logic.

These are the tests to read if you want to verify the semantic-modelling
approach itself rather than the figure it produces. Every step is exercised on
small inputs whose correct answer can be worked out by hand or by construction:

  * build_knn_graph     - honours the k cap and the similarity threshold;
  * association_strength - reproduces a_ij = w_ij * 2m / (s_i s_j) exactly;
  * detect_clusters      - recovers planted blocks and folds singletons;
  * label_clusters       - surfaces each block's characteristic vocabulary;
  * cluster_corpus       - is deterministic and covers every source.

The point of planting the structure (two dense blocks with no edges between
them) is that the partition is not a matter of opinion: a correct algorithm
must return exactly those two blocks, so a regression shows up as a hard
assertion failure rather than a subjective "looks worse".
"""
from __future__ import annotations

import numpy as np

from src import semantic_clusters as sc


def _two_block_similarity(block_size: int = 6) -> tuple[list, np.ndarray]:
    """Two dense blocks (intra-sim 0.9) with zero similarity between them."""
    n = 2 * block_size
    ids = [f"s{i}" for i in range(n)]
    sim = np.zeros((n, n))
    for start in (0, block_size):
        for i in range(start, start + block_size):
            for j in range(start, start + block_size):
                sim[i, j] = 1.0 if i == j else 0.9
    return ids, sim


def test_knn_graph_respects_threshold():
    ids, sim = _two_block_similarity()
    g = sc.build_knn_graph(ids, sim, k=10, threshold=0.5)
    # No edge may cross the two blocks (inter-block sim is 0 < threshold).
    for u, v in g.edges():
        assert (int(u[1:]) < 6) == (int(v[1:]) < 6)
    # Every node is present even if it had no qualifying neighbour.
    assert g.number_of_nodes() == len(ids)


def test_knn_graph_k_cap_actually_limits_edges():
    # Banded similarity: each node is above-threshold to several neighbours, so
    # the k cap genuinely bites. (A complete-graph test would pass even if the
    # cap were ignored, because all edges exist anyway.)
    n = 12
    ids = [f"s{i}" for i in range(n)]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            sim[i, j] = 1.0 if d == 0 else (1.0 - d * 0.12 if d <= 4 else 0.0)
    e2 = sc.build_knn_graph(ids, sim, k=2, threshold=0.5).number_of_edges()
    e4 = sc.build_knn_graph(ids, sim, k=4, threshold=0.5).number_of_edges()
    # If the cap were ignored these would be equal (all above-threshold edges).
    assert e2 < e4
    assert e2 < n * (n - 1) // 2  # not the complete graph


def test_association_strength_formula():
    import networkx as nx

    g = nx.Graph()
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("b", "c", weight=1.0)
    sc.association_strength(g)
    # weighted degrees: a=1, b=2, c=1 ; 2m = 4
    # assoc(a,b) = 1 * 4 / (1*2) = 2.0 ; assoc(b,c) = 1 * 4 / (2*1) = 2.0
    assert abs(g["a"]["b"]["assoc"] - 2.0) < 1e-9
    assert abs(g["b"]["c"]["assoc"] - 2.0) < 1e-9


def test_detect_clusters_recovers_planted_blocks():
    ids, sim = _two_block_similarity(block_size=6)
    g = sc.association_strength(sc.build_knn_graph(ids, sim, k=10, threshold=0.5))
    node2cluster, big_keys, modularity = sc.detect_clusters(g, resolution=1.0, seed=42, min_size=4)
    assert len(big_keys) == 2
    # Each planted block ends up wholly inside one cluster.
    block_a = {node2cluster[f"s{i}"] for i in range(6)}
    block_b = {node2cluster[f"s{i}"] for i in range(6, 12)}
    assert len(block_a) == 1 and len(block_b) == 1
    assert block_a != block_b
    assert modularity > 0.3  # two disconnected blocks -> strong modularity


def test_small_clusters_fold_into_other():
    # One block of 6 plus two isolated singletons -> singletons go to 'other'.
    ids, sim = _two_block_similarity(block_size=6)
    ids = ids[:6] + ["x", "y"]
    big = sim[:6, :6]
    sim2 = np.zeros((8, 8))
    sim2[:6, :6] = big
    g = sc.association_strength(sc.build_knn_graph(ids, sim2, k=10, threshold=0.5))
    node2cluster, big_keys, _ = sc.detect_clusters(g, min_size=4, seed=42)
    assert node2cluster["x"] == sc.OTHER_KEY
    assert node2cluster["y"] == sc.OTHER_KEY
    assert all(node2cluster[f"s{i}"] != sc.OTHER_KEY for i in range(6))


def test_label_clusters_surfaces_block_vocabulary():
    ids, sim = _two_block_similarity(block_size=6)
    texts = (["homomorphic encryption scheme"] * 6) + (["federated learning model"] * 6)
    res = sc.cluster_corpus(ids, texts, sim, k=10, threshold=0.5, resolution=1.0, seed=42, min_size=4)
    all_labels = " ".join(res.labels.values())
    assert any(w in all_labels for w in ("homomorphic", "encryption"))
    assert any(w in all_labels for w in ("federated", "learning"))
    # A label per big cluster, none empty.
    assert set(res.labels) == set(res.cluster_keys)
    assert all(res.labels[k] for k in res.cluster_keys)


def test_cluster_corpus_is_deterministic():
    ids, sim = _two_block_similarity()
    texts = [f"text {i}" for i in range(len(ids))]
    a = sc.cluster_corpus(ids, texts, sim, seed=42)
    b = sc.cluster_corpus(ids, texts, sim, seed=42)
    assert a.node2cluster == b.node2cluster
    assert a.cluster_keys == b.cluster_keys


# The real reproducibility hazard is Python hash randomisation changing set
# iteration order ACROSS processes (a same-process repeat cannot catch it).
_SUBPROC = """
import json, numpy as np
from src.semantic_clusters import cluster_corpus
bs = 6; n = 2 * bs
ids = [f"s{i}" for i in range(n)]
sim = np.zeros((n, n))
for start in (0, bs):
    for i in range(start, start + bs):
        for j in range(start, start + bs):
            sim[i, j] = 1.0 if i == j else 0.9
r = cluster_corpus(ids, [f"t{i}" for i in range(n)], sim, seed=42, min_size=4)
print(json.dumps(sorted(r.node2cluster.items())))
"""


def test_partition_is_deterministic_across_processes():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent

    def run(hashseed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        return subprocess.check_output(
            [sys.executable, "-c", _SUBPROC], cwd=str(root), env=env
        ).decode()

    # Different (and randomised) hash seeds must give an identical partition.
    assert run("0") == run("1") == run("random")


def test_cluster_corpus_covers_every_source_and_palettes_clusters():
    ids, sim = _two_block_similarity()
    texts = [f"text {i}" for i in range(len(ids))]
    res = sc.cluster_corpus(ids, texts, sim, seed=42, min_size=4)
    # Every id is assigned exactly one cluster key.
    assert set(res.node2cluster) == set(ids)
    # Cluster sizes sum to the number of sources.
    assert sum(res.sizes.values()) == len(ids)
    # Each big cluster (and 'other') has a colour.
    for key in res.cluster_keys:
        assert res.palette[key].startswith("#")
    assert res.palette[sc.OTHER_KEY] == sc.OTHER_COLOUR
    assert -1.0 <= res.modularity <= 1.0
