"""Stage 5 - topic assignment.

This module is deliberately hybrid. Fully unsupervised topic models (LDA,
Blei et al. 2003) would discover topics that need not match the structure of
the survey paper, defeating the point of the figure. Fully manual labelling
would be opaque to readers. We therefore:

    1. score each source against a *closed* set of topics defined in
       data/topic_vocabulary.yaml using TF-IDF cosine similarity between
       the source text and the topic keyword set (treated as a pseudo-document);
    2. surface the top-scoring topic and the runner-up to the human reviewer;
    3. record the final assignment in sources.csv:primary_topic, with the
       reviewer free to override the suggestion.

The scoring step is mechanical and reproducible. The override step is
explicit and auditable - every override is captured in the
`review_note` column. This mirrors the human-in-the-loop guidance in
PRISMA-S (Rethlefsen et al., 2021) on transparent reporting of inclusion
decisions in evidence syntheses.

PET-family tags are multi-label and are scored independently per family.
Any family whose normalised keyword count exceeds `family_threshold` is
attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TopicSuggestion:
    source_id: str
    primary_topic: str
    primary_score: float
    runner_up: str
    runner_up_score: float
    pet_families: list[str]
    margin: float

    def is_confident(self, min_margin: float = 0.05) -> bool:
        return self.margin >= min_margin


def _build_topic_corpus(vocab: dict) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    pseudo_docs: list[str] = []
    for t in vocab["topics"]:
        labels.append(t["label"])
        pseudo_docs.append(" ".join(t["keywords"]))
    return labels, pseudo_docs


def score_topics(
    source_ids: Iterable[str],
    source_texts: Iterable[str],
    vocab_path: Path,
) -> list[TopicSuggestion]:
    """Score every source against every topic and return ranked suggestions.

    Implementation note: rather than scoring "keyword present in text", we fit
    a single TF-IDF vectoriser over the union of (topic pseudo-documents,
    source texts) and take cosine similarity in that shared space. This
    avoids the bias of raw keyword-count scoring, where topics with longer
    keyword lists trivially win.
    """
    vocab = yaml.safe_load(vocab_path.read_text(encoding="utf-8"))
    topic_labels, topic_docs = _build_topic_corpus(vocab)
    pet_specs = vocab["pet_families"]

    ids = list(source_ids)
    texts = list(source_texts)

    combined = topic_docs + texts
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    matrix = vec.fit_transform(combined)
    topic_matrix = matrix[: len(topic_docs)]
    source_matrix = matrix[len(topic_docs):]
    sims = cosine_similarity(source_matrix, topic_matrix)

    out: list[TopicSuggestion] = []
    for i, source_id in enumerate(ids):
        scores = sims[i]
        if not np.any(scores):
            out.append(
                TopicSuggestion(
                    source_id=source_id,
                    primary_topic="",
                    primary_score=0.0,
                    runner_up="",
                    runner_up_score=0.0,
                    pet_families=[],
                    margin=0.0,
                )
            )
            continue
        ranked = np.argsort(scores)[::-1]
        primary_idx = int(ranked[0])
        runner_idx = int(ranked[1]) if len(ranked) > 1 else primary_idx
        families = _detect_families(texts[i], pet_specs)
        out.append(
            TopicSuggestion(
                source_id=source_id,
                primary_topic=topic_labels[primary_idx],
                primary_score=float(scores[primary_idx]),
                runner_up=topic_labels[runner_idx],
                runner_up_score=float(scores[runner_idx]),
                pet_families=families,
                margin=float(scores[primary_idx] - scores[runner_idx]),
            )
        )
    return out


def _detect_families(text: str, pet_specs: list[dict], threshold: int = 1) -> list[str]:
    if not text:
        return []
    # Normalise hyphens to spaces so that "zero-knowledge" and "zero knowledge"
    # are treated identically. The same normalisation is applied to keywords
    # for symmetry.
    lowered = text.lower().replace("-", " ")
    found: list[str] = []
    for spec in pet_specs:
        hits = 0
        for kw in spec["keywords"]:
            if kw.lower().replace("-", " ") in lowered:
                hits += 1
        if hits >= threshold:
            found.append(spec["label"])
    return found


def write_suggestions(suggestions: list[TopicSuggestion], path: Path) -> None:
    pd.DataFrame(
        [
            {
                "id": s.source_id,
                "suggested_primary_topic": s.primary_topic,
                "primary_score": round(s.primary_score, 4),
                "runner_up": s.runner_up,
                "runner_up_score": round(s.runner_up_score, 4),
                "margin": round(s.margin, 4),
                "confident": s.is_confident(),
                "suggested_pet_families": ";".join(s.pet_families),
            }
            for s in suggestions
        ]
    ).to_csv(path, index=False)
