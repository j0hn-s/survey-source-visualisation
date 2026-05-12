"""Shared fixtures for the test suite.

Tests focus on the deterministic parts of the pipeline: the reference parser,
the topic-assignment scoring, the graph edge rules, and the visualisation
smoke paths. Network-dependent stages (CrossRef / arXiv acquisition) and the
sentence-transformer embeddings are not covered here - they are integration
concerns, not unit-test material.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def vocab_path(project_root) -> Path:
    return project_root / "data" / "topic_vocabulary.yaml"


@pytest.fixture(scope="session")
def vocab(vocab_path):
    return yaml.safe_load(vocab_path.read_text(encoding="utf-8"))


@pytest.fixture()
def sample_metadata() -> pd.DataFrame:
    """A minimal but realistic metadata frame for graph and viz tests."""
    return pd.DataFrame(
        [
            {
                "id": "refA",
                "number": 1,
                "year": 2020,
                "authors": "Alice",
                "title": "A new DP composition theorem",
                "venue": "TCC",
                "publication_type": "academic_paper",
                "pet_family": "differential_privacy",
                "primary_topic": "mechanism",
            },
            {
                "id": "refB",
                "number": 2,
                "year": 2021,
                "authors": "Bob",
                "title": "Practical DP audit toolkit",
                "venue": "USENIX Security",
                "publication_type": "academic_paper",
                "pet_family": "differential_privacy",
                "primary_topic": "assurance",
            },
            {
                "id": "refC",
                "number": 3,
                "year": 2023,
                "authors": "Carol",
                "title": "Federated learning for healthcare",
                "venue": "Nature Medicine",
                "publication_type": "academic_paper",
                "pet_family": "federated_learning;differential_privacy",
                "primary_topic": "deployment",
            },
            {
                "id": "refD",
                "number": 4,
                "year": 2022,
                "authors": "Dave",
                "title": "ICO guidance on PETs",
                "venue": "ICO",
                "publication_type": "regulatory_guidance",
                "pet_family": "",
                "primary_topic": "governance",
            },
        ]
    )
