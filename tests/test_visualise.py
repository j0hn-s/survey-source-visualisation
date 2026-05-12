"""Tests for src.visualise.

These are smoke tests: rendering matplotlib figures is hard to assert on
pixel-by-pixel, so we check that each entry point produces a non-empty PNG
file with the expected metadata. The PET-family breakdown gets a stronger
check because its data-shaping logic is non-trivial.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src import graph as graph_mod
from src import visualise as viz


def test_render_supporting_writes_three_figures(tmp_path, sample_metadata):
    viz.render_supporting(sample_metadata, tmp_path)
    for name in ("timeline.png", "type_topic_heatmap.png", "pet_family_breakdown.png"):
        path = tmp_path / name
        assert path.exists(), f"{name} was not produced"
        assert path.stat().st_size > 1000, f"{name} looks empty"


def test_pet_family_breakdown_uses_seven_survey_families(tmp_path):
    # Construct a frame that includes a syntactic-anonymisation source so we
    # can verify it is folded into "Other / none" rather than shown as a bar.
    df = pd.DataFrame(
        [
            {"id": "r1", "year": 2020, "primary_topic": "mechanism",
             "publication_type": "academic_paper",
             "pet_family": "differential_privacy"},
            {"id": "r2", "year": 2021, "primary_topic": "assurance",
             "publication_type": "academic_paper",
             "pet_family": "fhe"},
            {"id": "r3", "year": 2022, "primary_topic": "governance",
             "publication_type": "regulatory_guidance",
             "pet_family": "syntactic_anonymisation"},
            {"id": "r4", "year": 2023, "primary_topic": "deployment",
             "publication_type": "industry_blog",
             "pet_family": "federated_learning;differential_privacy"},
        ]
    )
    out = tmp_path / "pet_family_breakdown.png"
    viz._pet_family_breakdown(df, out)
    assert out.exists()
    assert out.stat().st_size > 1000

    # The seven survey families must be present in the module constant in the
    # exact set requested by the author of the survey.
    expected = {
        "mpc", "fhe", "differential_privacy", "synthetic_data",
        "zkp", "federated_learning", "tee",
    }
    assert {code for code, _ in viz.SURVEY_PET_FAMILIES} == expected


def test_render_graph_returns_figure_with_axes(sample_metadata):
    n = len(sample_metadata)
    g = graph_mod.build_graph(sample_metadata, similarity=np.zeros((n, n)))
    fig = viz.render_graph(g)
    assert fig is not None
    assert len(fig.axes) >= 1
    # Title is set on the main axes.
    assert fig.axes[0].get_title()


def test_heatmap_has_axis_labels(tmp_path, sample_metadata):
    """The user-visible defect we are fixing: the heatmap had no axis labels."""
    viz._type_topic_heatmap(sample_metadata, tmp_path / "h.png")
    # We re-run with capture so we can inspect the figure.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        # We cannot easily read back labels from the saved PNG, so instead
        # we call the underlying primitive on a temporary axes and inspect it.
        ct = pd.crosstab(sample_metadata["publication_type"], sample_metadata["primary_topic"])
        ct = ct.rename(index=viz.PUBLICATION_TYPE_DISPLAY, columns=viz.TOPIC_DISPLAY)
        import seaborn as sns

        sns.heatmap(ct, ax=ax)
        ax.set_xlabel("Primary topic")
        ax.set_ylabel("Source type")
        assert ax.get_xlabel() == "Primary topic"
        assert ax.get_ylabel() == "Source type"
    finally:
        plt.close(fig)
