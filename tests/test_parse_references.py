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
    assert src.id == "ref001"
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
    # Every parsed source must have a stable id of the form refNNN.
    assert all(s.id.startswith("ref") and len(s.id) == 6 for s in sources)
    # Numbers should be a contiguous 1..N range (Harvard list is numbered).
    numbers = sorted(s.number for s in sources)
    assert numbers == list(range(1, len(sources) + 1))
