"""Stage 7 - rendering.

The main figure is a static matplotlib drawing because that is what the
paper accommodates. The layout is Fruchterman-Reingold (Fruchterman &
Reingold, 1991), which has been the de-facto default for bibliometric
network figures since CiteSpace and VOSviewer popularised it. We seed the
random state so that successive renderings produce the same picture.

We expose three rendering primitives:

    render_graph          - the main static figure
    render_interactive    - an HTML file (pyvis) for the appendix /
                            repository, useful for exploring without rerunning
    render_supporting     - timeline histogram, type x topic heatmap, and a
                            Sankey from publication type to topic to PET family
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


# Colour palettes deliberately use ColorBrewer-style qualitative schemes so
# that distinctions remain legible for colour-blind readers (Brewer, 1994).
TOPIC_PALETTE = {
    "mechanism": "#1f78b4",
    "systems": "#33a02c",
    "evaluation": "#ff7f00",
    "assurance": "#e31a1c",
    "governance": "#6a3d9a",
    "deployment": "#b15928",
    "survey": "#a6cee3",
    "": "#bdbdbd",
}

TYPE_MARKERS = {
    "academic_paper": "o",
    "survey_review": "D",
    "preprint": "s",
    "regulatory_guidance": "P",
    "technical_whitepaper": "^",
    "standards_specification": "X",
    "industry_blog": "v",
    "software_repository": "*",
    "": ".",
}

# Human-readable display names used on axis labels and legends.
PUBLICATION_TYPE_DISPLAY = {
    "academic_paper": "Academic paper",
    "survey_review": "Survey / SoK",
    "preprint": "Preprint",
    "regulatory_guidance": "Regulatory guidance",
    "technical_whitepaper": "Technical white paper",
    "standards_specification": "Standard / specification",
    "industry_blog": "Industry blog",
    "software_repository": "Software repository",
    "": "Unclassified",
}

TOPIC_DISPLAY = {
    "mechanism": "Mechanism",
    "systems": "Systems",
    "evaluation": "Evaluation",
    "assurance": "Assurance",
    "governance": "Governance",
    "deployment": "Deployment",
    "survey": "Survey",
    "": "Unclassified",
}

# The seven PET families the survey explicitly considers, in the order the
# author wants them displayed. `syntactic_anonymisation` is tracked in
# `sources.csv:pet_family` for completeness but is intentionally NOT shown in
# the PET-family breakdown figure - the figure answers "which of the seven
# survey-defined PETs do the sources focus on?" rather than "which anonymisation
# technique do they use?".
SURVEY_PET_FAMILIES: list[tuple[str, str]] = [
    ("mpc", "Secure multi-party computation"),
    ("fhe", "Homomorphic encryption"),
    ("differential_privacy", "Differential privacy"),
    ("synthetic_data", "Synthetic data"),
    ("zkp", "Zero-knowledge"),
    ("federated_learning", "Federated learning and distributed analytics"),
    ("tee", "Trusted execution environments"),
]


def _layout(g: nx.MultiGraph, seed: int = 42) -> dict:
    """Compute positions on a *weighted single-graph* projection of `g`.

    Topic edges are dense (every pair within a topic) and would collapse a
    spring layout into a single blob if weighted equally with semantic ones.
    We project the MultiGraph onto a simple weighted Graph where:

        semantic edges contribute their cosine-similarity weight directly;
        family   edges contribute a small constant attractive force;
        topic    edges contribute a very small attractive force.

    This matches the convention in VOSviewer (van Eck & Waltman, 2010) of
    using association strengths as spring weights rather than 0/1 edges.
    """
    simple = nx.Graph()
    simple.add_nodes_from(g.nodes(data=True))
    weights = {"semantic": 1.0, "family": 0.15, "topic": 0.05}
    for u, v, data in g.edges(data=True):
        w = weights.get(data.get("etype"), 0.0) * float(data.get("weight", 1.0))
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += w
        else:
            simple.add_edge(u, v, weight=w)
    return nx.spring_layout(
        simple,
        seed=seed,
        weight="weight",
        k=4.0 / np.sqrt(max(len(simple), 1)),
        iterations=400,
        scale=2.0,
    )


def render_graph(
    g: nx.MultiGraph,
    out_path: Optional[Path] = None,
    title: str = "Sources considered: semantic map",
    show_edges: Sequence[str] = ("topic", "semantic"),
    node_size_scale: float = 60.0,
    figsize: tuple[float, float] = (20, 14),
    seed: int = 42,
):
    """Render the main figure. Returns the matplotlib Figure for further tweaks.

    Defaults are tuned for a graph in the low-hundreds-of-nodes range, which
    is where the survey bibliography sits. Both `figsize` and `node_size_scale`
    are exposed so a smaller graph can be rendered without changing the call
    site. The legend is placed outside the plot area so it never overlaps the
    network when the layout spreads to the edges.
    """
    pos = _layout(g, seed=seed)
    fig, ax = plt.subplots(figsize=figsize)

    edge_styles = {
        "topic": {"alpha": 0.04, "width": 0.4, "edge_color": "#888888"},
        "family": {"alpha": 0.08, "width": 0.5, "edge_color": "#4d4d4d"},
        "semantic": {"alpha": 0.5, "width": 1.0, "edge_color": "#222222"},
    }
    for etype in show_edges:
        edgelist = [(u, v) for u, v, d in g.edges(data=True) if d.get("etype") == etype]
        if edgelist:
            nx.draw_networkx_edges(
                g,
                pos,
                edgelist=edgelist,
                ax=ax,
                **edge_styles.get(etype, {}),
            )

    degrees = dict(g.degree())
    grouped: dict[tuple[str, str], list[str]] = {}
    for node, data in g.nodes(data=True):
        key = (data.get("primary_topic") or "", data.get("publication_type") or "")
        grouped.setdefault(key, []).append(node)

    for (topic, ptype), nodes in grouped.items():
        x = [pos[n][0] for n in nodes]
        y = [pos[n][1] for n in nodes]
        sizes = [node_size_scale + 14 * degrees.get(n, 0) for n in nodes]
        ax.scatter(
            x,
            y,
            s=sizes,
            c=TOPIC_PALETTE.get(topic, "#bdbdbd"),
            marker=TYPE_MARKERS.get(ptype, "."),
            edgecolors="white",
            linewidths=0.6,
            label=None,
        )

    _draw_legends(ax)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_axis_off()
    # Crop the visible area to the bulk of nodes (5th-95th percentile in each
    # dimension) so that a handful of disconnected outliers - which spring
    # layout repels far from the main mass - do not dictate the framing.
    # Outlier nodes are still drawn; they will sit just outside the visible
    # window. Cropping is purely a viewport decision.
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    if xs.size and ys.size:
        x_lo, x_hi = np.percentile(xs, [5, 95])
        y_lo, y_hi = np.percentile(ys, [5, 95])
        x_pad = 0.08 * (x_hi - x_lo or 1.0)
        y_pad = 0.08 * (y_hi - y_lo or 1.0)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    fig.subplots_adjust(left=0.03, right=0.78, top=0.94, bottom=0.04)
    if out_path:
        # Avoid bbox_inches="tight" so the requested figsize is preserved.
        fig.savefig(out_path, dpi=200)
    return fig


def _draw_legends(ax) -> None:
    topic_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=10, label=TOPIC_DISPLAY.get(t, t))
        for t, c in TOPIC_PALETTE.items() if t
    ]
    type_handles = [
        plt.Line2D([0], [0], marker=m, color="#444", linestyle="",
                   markersize=10, label=PUBLICATION_TYPE_DISPLAY.get(t, t))
        for t, m in TYPE_MARKERS.items() if t
    ]
    leg1 = ax.legend(
        handles=topic_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Primary topic",
        frameon=False,
        fontsize=10,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=type_handles,
        bbox_to_anchor=(1.02, 0.55),
        loc="upper left",
        title="Source type",
        frameon=False,
        fontsize=10,
    )


def render_interactive(g: nx.MultiGraph, out_path: Path) -> None:
    from pyvis.network import Network  # type: ignore

    net = Network(height="800px", width="100%", bgcolor="#ffffff", notebook=False)
    for node, data in g.nodes(data=True):
        net.add_node(
            node,
            label=data.get("label") or node,
            title=_node_tooltip(data),
            color=TOPIC_PALETTE.get(data.get("primary_topic") or "", "#bdbdbd"),
            shape="dot",
        )
    for u, v, data in g.edges(data=True):
        if data.get("etype") == "semantic":
            net.add_edge(u, v, value=data.get("weight", 0.5), title=data.get("etype"))
    net.write_html(str(out_path), open_browser=False)


def _node_tooltip(data: dict) -> str:
    return (
        f"<b>{data.get('label','')}</b><br>"
        f"{data.get('title','')}<br>"
        f"Type: {data.get('publication_type','')}<br>"
        f"Topic: {data.get('primary_topic','')}<br>"
        f"Families: {data.get('pet_family','')}"
    )


def render_supporting(metadata: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _timeline(metadata, out_dir / "timeline.png")
    _type_topic_heatmap(metadata, out_dir / "type_topic_heatmap.png")
    _pet_family_breakdown(metadata, out_dir / "pet_family_breakdown.png")


def _timeline(metadata: pd.DataFrame, out_path: Path) -> None:
    df = metadata.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    counts = (
        df.groupby(["year", "primary_topic"]).size().reset_index(name="n")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = counts.pivot(index="year", columns="primary_topic", values="n").fillna(0)
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[TOPIC_PALETTE.get(c, "#bdbdbd") for c in pivot.columns],
    )
    ax.set_ylabel("Sources")
    ax.set_xlabel("Year")
    ax.set_title("Sources by year and primary topic")
    ax.legend(title="Primary topic", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _type_topic_heatmap(metadata: pd.DataFrame, out_path: Path) -> None:
    """Source type × primary topic crosstab, with human-readable axes."""
    ct = pd.crosstab(metadata["publication_type"], metadata["primary_topic"])
    ct = ct.rename(
        index=PUBLICATION_TYPE_DISPLAY,
        columns=TOPIC_DISPLAY,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        ax=ax,
        cbar_kws={"label": "Number of sources"},
    )
    ax.set_title("Sources by source type and primary topic")
    ax.set_xlabel("Primary topic")
    ax.set_ylabel("Source type")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pet_family_breakdown(metadata: pd.DataFrame, out_path: Path) -> None:
    """PET-family count, stacked by primary topic.

    Each source can carry multiple PET-family tags. A source that is tagged
    `differential_privacy;federated_learning` contributes one count to both
    families' bars. The total across bars therefore exceeds the number of
    sources, which is intentional - the figure answers "how many sources
    consider each PET?", not "how do the sources partition?".

    Restricted to the seven PET families the survey explicitly defines
    (see SURVEY_PET_FAMILIES). Sources that carry no tag from the seven are
    summarised as a separate "Other / none" bar so they are still visible.
    """
    family_lookup = dict(SURVEY_PET_FAMILIES)
    family_order = [code for code, _ in SURVEY_PET_FAMILIES]
    topic_order = [t for t in TOPIC_PALETTE if t] + [""]

    rows: list[dict] = []
    n_outside = 0
    for _, row in metadata.iterrows():
        families = [
            f.strip()
            for f in str(row.get("pet_family") or "").split(";")
            if f.strip()
        ]
        kept = [f for f in families if f in family_lookup]
        topic = row.get("primary_topic") or ""
        if kept:
            for f in kept:
                rows.append({"family": f, "topic": topic})
        else:
            n_outside += 1

    breakdown = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6.5))

    if breakdown.empty and n_outside == 0:
        ax.text(
            0.5, 0.5,
            "No sources carry any of the seven survey PET-family tags.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    if breakdown.empty:
        pivot = pd.DataFrame(0, index=family_order, columns=[""])
    else:
        pivot = (
            breakdown.groupby(["family", "topic"]).size().unstack(fill_value=0)
            .reindex(family_order, fill_value=0)
        )
        pivot = pivot.reindex(
            columns=[t for t in topic_order if t in pivot.columns], fill_value=0
        )
    pivot.index = [family_lookup[c] for c in pivot.index]

    # Add an "Other / none" row carrying the count of sources that did not
    # carry any of the seven survey families. Stack it in a synthetic column
    # so it shares the same y-axis and rendering call as the rest.
    if n_outside:
        pivot["Other / none"] = 0
        pivot.loc["Other / none"] = 0
        pivot.loc["Other / none", "Other / none"] = n_outside

    colours = [
        "#bdbdbd" if c == "Other / none" else TOPIC_PALETTE.get(c, "#bdbdbd")
        for c in pivot.columns
    ]
    pivot.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=colours,
        edgecolor="white",
        linewidth=0.6,
        width=0.75,
    )

    ax.invert_yaxis()  # most-discussed PET on top
    ax.set_xlabel(
        "Number of sources (multi-label: a source may contribute to several bars)"
    )
    ax.set_ylabel("PET family")
    ax.set_title("Sources by PET family, stacked by primary topic")
    legend_labels = [
        "Outside seven-family taxonomy" if c == "Other / none" else TOPIC_DISPLAY.get(c, c)
        for c in pivot.columns
    ]
    ax.legend(
        legend_labels,
        title="Primary topic",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
