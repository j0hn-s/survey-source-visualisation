"""Stage 5 (revised) - data-driven semantic clustering (science mapping).

This module replaces the a-priori controlled-vocabulary topic *colour* with
EMERGENT clusters detected on the source-similarity network. It follows the
established science-mapping methodology rather than inventing one:

  1. Sparsify the cosine-similarity matrix to each source's k nearest
     neighbours - the association-cap step VOSviewer uses to stop a dense
     similarity graph collapsing into a hairball (van Eck & Waltman, 2010,
     Software survey: VOSviewer, Scientometrics 84(2)).

  2. Re-weight every surviving edge to its ASSOCIATION STRENGTH,
     a_ij = w_ij * (2m) / (s_i * s_j), where s_i is the weighted degree of i
     and 2m the total edge weight. This is van Eck & Waltman's normalisation
     (also called the proximity / probabilistic-affinity index): it corrects
     for the fact that a source similar to many others co-occurs strongly by
     chance, so raw similarity over-states its importance.

  3. Partition the normalised network by modularity maximisation using the
     Louvain method (Blondel, Guillaume, Lambiotte & Lefebvre, 2008, Fast
     unfolding of communities in large networks, J. Stat. Mech.). The number
     of clusters is NOT chosen by us; it emerges from the structure. A fixed
     seed makes the partition reproducible.

  4. Label each cluster by its most characteristic terms via class-based
     TF-IDF: concatenate every title in a cluster into one pseudo-document and
     rank terms by TF-IDF across clusters. This is the classical analogue of
     the c-TF-IDF labelling popularised by BERTopic (Grootendorst, 2022).

Why emergent clustering and not the controlled-vocabulary scorer in
`topic_assignment.py`? Scoring 8-to-12-word titles against keyword bags yields
near-tie margins (about 86% of sources fall below the methodology's own 0.05
confidence bar), so an a-priori topic colour is not trustworthy as a headline
encoding. Letting the corpus define its own clusters is exactly the claim a
"semantic map of the sources considered" should be able to support. The
controlled vocabulary is retained for the human-review workflow and the PET
family tags; it is simply no longer the basis of the figure's colour.

Everything here is deterministic given (similarity matrix, k, threshold,
resolution, seed), which is what makes the approach testable - see
`tests/test_semantic_clusters.py`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from networkx.algorithms import community as nx_comm
from sklearn.feature_extraction.text import TfidfVectorizer

# Colour-blind-safe qualitative palette. This is Paul Tol's "muted" scheme
# (Tol, 2021, "Colour schemes", SRON technical note SRON/EPS/TN/09-002),
# designed to stay distinguishable under deuteranopia, protanopia and
# tritanopia - the right choice for up to ~9 categorical classes. Indexed by
# emergent-cluster rank (largest cluster first); 'other' uses Tol's pale grey.
_CLUSTER_COLOURS = [
    "#332288",  # indigo            (Tol muted)
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
    # Overflow (Tol "vibrant"), so a partition with >9 big clusters does not
    # wrap onto a reused colour. The current build produces 9.
    "#EE7733",  # orange
    "#0077BB",  # blue
    "#EE3377",  # magenta
]
OTHER_KEY = "other"
OTHER_COLOUR = "#DDDDDD"


@dataclass
class ClusterResult:
    node2cluster: dict           # source id -> cluster key ("C1".. or "other")
    cluster_keys: list           # ordered big-cluster keys, largest first
    labels: dict                 # cluster key -> "term, term, term"
    palette: dict                # cluster key -> hex colour
    sizes: dict                  # cluster key -> node count
    modularity: float            # modularity of the returned partition
    graph: nx.Graph = field(repr=False, default=None)  # assoc-weighted graph

    def display(self) -> dict:
        """cluster key -> legend label ('C1: term, term, term')."""
        out = {}
        for k in self.cluster_keys:
            out[k] = f"{k}: {self.labels.get(k, '')}"
        out[OTHER_KEY] = "Other / sparsely connected"
        return out


def build_knn_graph(ids, sim: np.ndarray, k: int = 8, threshold: float = 0.15) -> nx.Graph:
    """Undirected k-NN cosine-similarity graph. All ids are nodes (isolated
    sources stay in the graph as degree-0 nodes so the partition covers them)."""
    g = nx.Graph()
    g.add_nodes_from(ids)
    n = sim.shape[0]
    for i in range(n):
        row = sim[i].copy()
        row[i] = -1.0
        for j in np.argsort(row)[::-1][:k]:
            j = int(j)
            w = float(sim[i, j])
            if w < threshold or i == j:
                continue
            a, b = ids[i], ids[j]
            if g.has_edge(a, b):
                if w > g[a][b]["weight"]:
                    g[a][b]["weight"] = w
            else:
                g.add_edge(a, b, weight=w)
    return g


def association_strength(g: nx.Graph) -> nx.Graph:
    """Annotate every edge with van Eck & Waltman association strength `assoc`.

    a_ij = w_ij * (2m) / (s_i * s_j). Mutates and returns `g`.
    """
    strength = dict(g.degree(weight="weight"))
    two_m = float(sum(strength.values()))
    for u, v, d in g.edges(data=True):
        denom = strength[u] * strength[v]
        d["assoc"] = (d["weight"] * two_m / denom) if denom > 0 else 0.0
    return g


def detect_clusters(
    g: nx.Graph, resolution: float = 1.0, seed: int = 42, min_size: int = 4
) -> tuple[dict, list, float]:
    """Louvain partition on association strength. Clusters smaller than
    `min_size` are folded into a single 'other' bucket. Returns
    (node2cluster, ordered_big_keys, modularity).

    Louvain's tie-breaking iterates over sets of node labels, whose order under
    string keys varies per process with hash randomisation - which would make
    the partition non-reproducible across runs even with a fixed RNG seed. We
    therefore relabel the nodes to integers in a fixed (sorted) order before
    partitioning, so the result depends only on (graph, resolution, seed).
    """
    h = nx.convert_node_labels_to_integers(g, ordering="sorted", label_attribute="_orig")
    int2orig = {i: h.nodes[i]["_orig"] for i in h.nodes}
    int_communities = nx_comm.louvain_communities(
        h, weight="assoc", resolution=resolution, seed=seed
    )
    communities = [{int2orig[i] for i in comm} for comm in int_communities]
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))
    modularity = float(nx_comm.modularity(g, communities, weight="assoc"))

    node2cluster: dict = {}
    big_keys: list = []
    rank = 0
    for comm in communities:
        if len(comm) >= min_size:
            rank += 1
            key = f"C{rank}"
            big_keys.append(key)
            for nid in comm:
                node2cluster[nid] = key
        else:
            for nid in comm:
                node2cluster[nid] = OTHER_KEY
    return node2cluster, big_keys, modularity


def label_clusters(node2cluster, ids, texts, big_keys, top_n: int = 3) -> dict:
    """Characteristic terms per cluster via class-based TF-IDF over the
    cluster pseudo-documents."""
    idx = {sid: i for i, sid in enumerate(ids)}
    bag: dict = defaultdict(list)
    for sid, key in node2cluster.items():
        if key in big_keys and sid in idx:
            bag[key].append(texts[idx[sid]])

    if not big_keys:
        return {}
    corpus = [" ".join(bag[k]) for k in big_keys]
    # max_df drops terms common to most clusters ("data", "privacy") which are
    # not discriminative; this is what makes c-TF-IDF labels specific. Fall
    # back to no upper cut if that prunes everything (very few clusters).
    try:
        vec = TfidfVectorizer(min_df=1, max_df=0.6, ngram_range=(1, 3), stop_words="english")
        matrix = vec.fit_transform(corpus)
    except ValueError:
        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 3), stop_words="english")
        matrix = vec.fit_transform(corpus)
    terms = np.array(vec.get_feature_names_out())
    term_tokens = [t.split() for t in terms]
    labels: dict = {}
    for r, key in enumerate(big_keys):
        weights = matrix[r].toarray().ravel()
        # Mildly prefer multi-word phrases at comparable weight so a coherent
        # phrase ("zero knowledge proofs") beats its constituent unigrams.
        adjusted = weights * np.array([1.0 + 0.2 * (len(t) - 1) for t in term_tokens])
        order = np.argsort(adjusted)[::-1]
        chosen: list = []
        used_tokens: set = set()
        for t_idx in order:
            if weights[t_idx] <= 0:
                break
            tokens = set(term_tokens[t_idx])
            # Keep the label to distinct concepts: skip any term that shares a
            # word with one already chosen, so we get "zero knowledge proofs,
            # halo2, gpu" rather than "zero knowledge, knowledge proofs".
            if tokens & used_tokens:
                continue
            chosen.append(terms[t_idx])
            used_tokens |= tokens
            if len(chosen) >= top_n:
                break
        labels[key] = ", ".join(chosen)
    return labels


def cluster_corpus(
    ids, texts, sim: np.ndarray,
    k: int = 8, threshold: float = 0.15, resolution: float = 1.0,
    seed: int = 42, min_size: int = 4, top_n: int = 3,
) -> ClusterResult:
    """Full pipeline: k-NN graph -> association strength -> Louvain -> labels."""
    g = association_strength(build_knn_graph(ids, sim, k=k, threshold=threshold))
    node2cluster, big_keys, modularity = detect_clusters(
        g, resolution=resolution, seed=seed, min_size=min_size
    )
    labels = label_clusters(node2cluster, ids, texts, big_keys, top_n=top_n)
    palette = {key: _CLUSTER_COLOURS[i % len(_CLUSTER_COLOURS)] for i, key in enumerate(big_keys)}
    palette[OTHER_KEY] = OTHER_COLOUR
    sizes: dict = defaultdict(int)
    for key in node2cluster.values():
        sizes[key] += 1
    return ClusterResult(
        node2cluster=node2cluster, cluster_keys=big_keys, labels=labels,
        palette=palette, sizes=dict(sizes), modularity=modularity, graph=g,
    )
