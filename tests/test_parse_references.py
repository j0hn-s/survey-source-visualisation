"""Tests for src.parse_references.

These pin the parser against representative entries drawn from
`reference_list.txt`. The aim is to catch regressions in the field-extraction
regexes - especially around typographic quotes, year-with-suffix patterns
(e.g. `2025a`), and DOI vs. plain URL handling.
"""

from __future__ import annotations

from src import parse_references as pr


def test_parses_typical_doi_entry(vocab):
    entry = (
        "1.\tAbadi, M., Chu, A., Goodfellow, I., McMahan, H.B., Mironov, I., "
        "Talwar, K. and Zhang, L. (2016) 'Deep learning with differential privacy', "
        "in Proceedings of the 2016 ACM SIGSAC Conference on Computer and "
        "Communications Security. New York: ACM, pp. 308-318. "
        "<https://doi.org/10.1145/2976749.2978318>"
    )
    src = pr.parse_entry(entry, vocab)
    assert src is not None
    assert src.number == 1
    # Content-derived id; we do not pin the exact hash but require the format.
    assert src.id.startswith("ref-") and len(src.id) == 12
    assert src.year == 2016
    assert src.title == "Deep learning with differential privacy"
    assert src.url == "https://doi.org/10.1145/2976749.2978318"
    assert src.doi == "10.1145/2976749.2978318"
    assert src.authors.startswith("Abadi, M.")
    # Conference paper -> academic_paper -> abstract acquisition strategy.
    assert src.publication_type_seed == "academic_paper"
    assert src.text_strategy == "abstract"
    assert "differential_privacy" in src.pet_family_seed


def test_parses_arxiv_preprint(vocab):
    entry = (
        "69.\tHard, A., Rao, K., Mathews, R. (2018) "
        "'Federated learning for mobile keyboard prediction', "
        "arXiv preprint arXiv:1811.03604. "
        "Available at: <https://arxiv.org/abs/1811.03604>"
    )
    src = pr.parse_entry(entry, vocab)
    assert src is not None
    assert src.year == 2018
    assert src.title == "Federated learning for mobile keyboard prediction"
    assert src.arxiv_id == "1811.03604"
    assert src.publication_type_seed == "preprint"
    assert src.text_strategy == "abstract"
    assert "federated_learning" in src.pet_family_seed


def test_parses_year_with_suffix(vocab):
    entry = (
        "142.\tNational Quantum Computing Centre (2025a) 'Quantum computing use "
        "case compendium'. Available at: <https://example.org/compendium.pdf>"
    )
    src = pr.parse_entry(entry, vocab)
    assert src is not None
    assert src.year == 2025


def test_publication_type_pattern_priority(vocab):
    # An ISO entry should resolve to standards_specification even though it
    # also contains the word "Information".
    entry = (
        "92.\tISO/IEC (2018) ISO/IEC 20889:2018 Privacy enhancing data "
        "de-identification - Terminology and classification. "
        "Geneva: International Organization for Standardization."
    )
    src = pr.parse_entry(entry, vocab)
    assert src is not None
    assert src.publication_type_seed == "standards_specification"
    assert src.text_strategy == "summary"


def test_software_repository_routes_to_manual(vocab):
    entry = (
        "76.\tIBM (2025) HElayers: A software development kit for homomorphic "
        "encryption-based analytics and machine learning [Software]. "
        "GitHub repository. Available at: <https://github.com/IBM/helayers>"
    )
    src = pr.parse_entry(entry, vocab)
    assert src is not None
    assert src.publication_type_seed == "software_repository"
    assert src.text_strategy == "manual"


def test_blank_line_is_ignored(vocab):
    assert pr.parse_entry("", vocab) is None
    assert pr.parse_entry("   \t   ", vocab) is None


def test_full_file_round_trip(project_root, vocab):
    """Sanity check that the live reference list parses cleanly end-to-end."""
    sources = pr.parse_file(project_root / "reference_list.txt", vocab)
    assert len(sources) > 200, "expected the full bibliography (>200 entries)"
    # Every parsed source carries a content-derived id of the form ref-XXXXXXXX.
    assert all(s.id.startswith("ref-") and len(s.id) == 12 for s in sources)
    # Numbers should be a contiguous 1..N range (Harvard list is numbered).
    numbers = sorted(s.number for s in sources)
    assert numbers == list(range(1, len(sources) + 1))


def test_id_is_stable_under_renumbering(vocab):
    """A reference's id depends only on its content, not its position."""
    body = (
        "Author, A. (2024) 'Paper title', Venue. "
        "<https://doi.org/10.1234/xyz>"
    )
    a = pr.parse_entry("1.\t" + body, vocab)
    b = pr.parse_entry("99.\t" + body, vocab)
    assert a is not None and b is not None
    assert a.id == b.id
    # Position changes, id does not.
    assert a.number == 1
    assert b.number == 99


def test_id_is_stable_under_whitespace_variation(vocab):
    """Trailing whitespace and tab vs space must not change the id."""
    a = pr.parse_entry("1.\tAuthor (2024) 'Title', Venue.", vocab)
    b = pr.parse_entry("1.   Author  (2024)   'Title',  Venue.   ", vocab)
    assert a is not None and b is not None
    assert a.id == b.id


def test_id_changes_when_substantive_text_changes(vocab):
    """A typo fix changes the id - intentional, surfaces an explicit migration."""
    a = pr.parse_entry("1.\tAuthor (2024) 'Title A', Venue.", vocab)
    b = pr.parse_entry("1.\tAuthor (2024) 'Title B', Venue.", vocab)
    assert a is not None and b is not None
    assert a.id != b.id


def test_merge_preserves_curation_for_existing_ids(vocab):
    parsed = [
        pr.parse_entry("1.\tAuthor (2024) 'Stable paper', Venue.", vocab),
    ]
    assert parsed[0] is not None
    existing = {
        parsed[0].id: {
            "id": parsed[0].id,
            "publication_type": parsed[0].publication_type_seed,
            "publication_type_seed": parsed[0].publication_type_seed,
            "pet_family": "",
            "pet_family_seed": "",
            "primary_topic": "mechanism",
            "secondary_topics": "",
            "review_note": "reviewed 2026-05",
        }
    }
    rows, report = pr.merge_sources(parsed, existing)
    assert len(rows) == 1
    assert rows[0]["primary_topic"] == "mechanism"
    assert rows[0]["review_note"] == "reviewed 2026-05"
    assert report.curated_preserved == 1
    assert report.added_ids == []
    assert report.removed_ids == []


def test_merge_flags_added_and_removed(vocab):
    parsed = [
        pr.parse_entry("1.\tAuthor (2024) 'New paper', Venue.", vocab),
    ]
    assert parsed[0] is not None
    # An old entry that no longer appears in reference_list.txt.
    existing = {
        "ref-deadbeef": {
            "id": "ref-deadbeef",
            "publication_type": "academic_paper",
            "publication_type_seed": "academic_paper",
            "pet_family": "",
            "pet_family_seed": "",
            "primary_topic": "mechanism",
            "secondary_topics": "",
            "review_note": "previously curated",
        }
    }
    _, report = pr.merge_sources(parsed, existing)
    assert parsed[0].id in report.added_ids
    assert "ref-deadbeef" in report.removed_ids
    assert report.curated_dropped == 1  # the removed entry had curation


def test_merge_dedupes_identical_entries(vocab):
    body = "Author (2024) 'Duplicate paper', Venue."
    parsed = [
        pr.parse_entry(f"{i}.\t" + body, vocab) for i in (1, 2, 3)
    ]
    assert all(p is not None for p in parsed)
    rows, report = pr.merge_sources(parsed, existing={})
    assert len(rows) == 1
    assert len(report.duplicates) == 1
    sid, numbers = report.duplicates[0]
    assert numbers == [1, 2, 3]
    assert sid == parsed[0].id


def test_has_curation_ignores_seed_carryover():
    """A row whose publication_type equals its seed has not been curated."""
    row = {
        "publication_type": "academic_paper",
        "publication_type_seed": "academic_paper",
        "pet_family": "",
        "pet_family_seed": "",
        "primary_topic": "",
        "secondary_topics": "",
        "review_note": "",
    }
    assert pr._has_curation(row) is False
    row["primary_topic"] = "mechanism"
    assert pr._has_curation(row) is True


def test_has_curation_detects_override_of_seed():
    row = {
        "publication_type": "survey_review",          # overridden
        "publication_type_seed": "academic_paper",     # seed
        "pet_family": "",
        "pet_family_seed": "",
        "primary_topic": "",
        "secondary_topics": "",
        "review_note": "",
    }
    assert pr._has_curation(row) is True
