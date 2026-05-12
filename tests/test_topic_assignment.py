"""Tests for src.topic_assignment.

We construct deliberately crafted texts whose topical home is unambiguous
and check that the scoring routine surfaces the expected primary topic and
correctly detects the PET family. Borderline texts (low margin) are also
exercised so the `confident` flag is meaningful.
"""

from __future__ import annotations

from src import topic_assignment as ta


def test_mechanism_text_routes_to_mechanism(vocab_path):
    texts = [
        "We introduce a new differential privacy mechanism and prove "
        "composition bounds via Renyi divergence. Lemma 1 establishes "
        "sensitivity calibration; Theorem 2 proves tightness.",
    ]
    suggestions = ta.score_topics(["t1"], texts, vocab_path)
    assert suggestions[0].primary_topic == "mechanism"
    assert "differential_privacy" in suggestions[0].pet_families


def test_governance_text_routes_to_governance(vocab_path):
    texts = [
        "This regulatory guidance from the Information Commissioner provides "
        "a compliance framework for PETs under the GDPR. We recommend that "
        "organisations follow these standards.",
    ]
    suggestions = ta.score_topics(["t2"], texts, vocab_path)
    assert suggestions[0].primary_topic == "governance"


def test_evaluation_text_routes_to_evaluation(vocab_path):
    texts = [
        "We benchmark throughput, latency, and communication overhead of "
        "three zero-knowledge proof systems on a comparative evaluation suite.",
    ]
    suggestions = ta.score_topics(["t3"], texts, vocab_path)
    assert suggestions[0].primary_topic == "evaluation"
    assert "zkp" in suggestions[0].pet_families


def test_empty_text_produces_blank_suggestion(vocab_path):
    suggestions = ta.score_topics(["t4"], [""], vocab_path)
    s = suggestions[0]
    assert s.primary_topic == ""
    assert s.primary_score == 0.0
    assert s.is_confident() is False


def test_low_margin_is_not_confident(vocab_path):
    # Text deliberately balanced between mechanism and survey wording.
    texts = ["a survey of mechanisms"]
    suggestions = ta.score_topics(["t5"], texts, vocab_path)
    s = suggestions[0]
    # We do not pin the chosen topic - we only require that the margin is
    # honest about how close the call is.
    assert s.margin < 0.5
    if s.margin < 0.05:
        assert s.is_confident() is False


def test_write_suggestions_round_trips(tmp_path, vocab_path):
    texts = ["a comprehensive survey of secure multi-party computation"]
    suggestions = ta.score_topics(["t6"], texts, vocab_path)
    out = tmp_path / "suggestions.csv"
    ta.write_suggestions(suggestions, out)
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "t6" in body
    assert "suggested_primary_topic" in body
