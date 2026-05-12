"""Stage 6 - graph construction.

Edges are produced by three independent rules so that the visualisation can
toggle between them and the reader can see what each rule contributes:

    topic     - two sources share a primary topic. Equivalent to the cluster
                colouring; included as edges so that single-topic clusters
                still produce visible connective tissue.

    family    - two sources share at least one PET family tag. This is a
                bibliographic-coupling-style relation (Kessler, 1963):
                co-membership in a topical class implies relatedness even if
                the texts do not look similar.

    semantic  - cosine similarity above a chosen threshold on either the
                TF-IDF or the sentence-embedding representation. We use a
                k-nearest-neighbour cap (default k=5) so that the graph does
                not collapse into a hairball. The k-NN choice is borrowed
                from VOSviewer (van Eck & Waltman, 2010, §3.2) which uses an
                analogous cap on association strengths.

The graph object is a `networkx.MultiGraph` whose edges carry an `etype`
attribute. Downstream rendering filters on this attribute.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(
    metadata: pd.DataFrame,
    similarity: np.ndarray,
    k_semantic: int = 5,
    semantic_threshold: float = 0.25,
) -> nx.MultiGraph:
    g = nx.MultiGraph()

    for _, row in metadata.iterrows():
        g.add_node(
            row["id"],
            label=_short_label(row),
            year=_safe_year(row.get("year")),
            publication_type=row.get("publication_type") or "",
            primary_topic=row.get("primary_topic") or "",
            pet_family=row.get("pet_family") or "",
            title=row.get("title") or "",
            authors=row.get("authors") or "",
        )

    ids = metadata["id"].tolist()
    id_to_idx = {sid: i for i, sid in enumerate(ids)}

    _add_topic_edges(g, metadata)
    _add_family_edges(g, metadata)
    if similarity is not None and similarity.size:
        _add_semantic_edges(
            g, ids, id_to_idx, similarity, k_semantic, semantic_threshold
        )
    return g


def _safe_year(value) -> int | None:
    if value is None or value == "" or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _short_label(row: pd.Series) -> str:
    authors = (row.get("authors") or "").split(",")
    first = authors[0].strip() if authors else ""
    year = _safe_year(row.get("year"))
    year_str = str(year) if year is not None else ""
    return f"{first} {year_str}".strip()


def _add_topic_edges(g: nx.MultiGraph, metadata: pd.DataFrame) -> None:
    for topic, group in metadata.groupby("primary_topic"):
        if not topic:
            continue
        ids = group["id"].tolist()
        for a, b in _pairs(ids):
            g.add_edge(a, b, etype="topic", weight=1.0)


def _add_family_edges(g: nx.MultiGraph, metadata: pd.DataFrame) -> None:
    family_to_ids: dict[str, list[str]] = {}
    for _, row in metadata.iterrows():
        fams = [f.strip() for f in (row.get("pet_family") or "").split(";") if f.strip()]
        for f in fams:
            family_to_ids.setdefault(f, []).append(row["id"])
    for fam, ids in family_to_ids.items():
        for a, b in _pairs(ids):
            g.add_edge(a, b, etype="family", weight=1.0, family=fam)


def _add_semantic_edges(
    g: nx.MultiGraph,
    ids: list[str],
    id_to_idx: dict[str, int],
    sim: np.ndarray,
    k: int,
    threshold: float,
) -> None:
    n = sim.shape[0]
    for i in range(n):
        row = sim[i].copy()
        row[i] = -1.0
        nbrs = np.argsort(row)[::-1][:k]
        for j in nbrs:
            if sim[i, j] < threshold:
                continue
            a, b = ids[i], ids[int(j)]
            # Use unordered pair to avoid double-counting.
            if a >= b:
                continue
            g.add_edge(a, b, etype="semantic", weight=float(sim[i, int(j)]))


def _pairs(items: Iterable[str]):
    items = list(items)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def filter_edges(g: nx.MultiGraph, etypes: Iterable[str]) -> nx.MultiGraph:
    """Return a view of the graph keeping only edges with `etype` in `etypes`."""
    keep = set(etypes)
    out = nx.MultiGraph()
    out.add_nodes_from(g.nodes(data=True))
    for u, v, data in g.edges(data=True):
        if data.get("etype") in keep:
            out.add_edge(u, v, **data)
    return out
