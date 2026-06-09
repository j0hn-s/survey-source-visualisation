# Data dictionary — `data/sources.csv`

One row per cited source. The CSV is the audited source of truth: every
downstream artefact (text payloads, suggestions, the graph, the figure) is
keyed on `id` and reproducible from this file.

Fields are grouped by provenance.

## Mechanical fields (set by the parser, never edited by hand)

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable, content-derived identifier of the form `ref-XXXXXXXX` (8 hex chars of SHA-1 over the normalised raw entry). Does not change when the bibliography is renumbered or when entries are inserted; does change if the entry's substantive text is edited. |
| `number` | int | Position in the bibliography as given by the author. Mutable across re-parses; use `id` for lookups. |
| `raw` | string | The verbatim bibliography entry. Kept so a reviewer can always trace a row back to its source. |
| `authors` | string | Text before the first `(YYYY)` token. |
| `year` | int or blank | First four-digit year inside parentheses. Blank when none is detected; flagged for manual review. |
| `title` | string | Text between the first pair of typographic quotes. |
| `venue` | string | Remainder after the title and before the URL. |
| `url` | string | First `<https://…>` URL detected, if any. |
| `doi` | string | DOI suffix detected on the URL (e.g. `10.1145/3133956.3133982`). |
| `arxiv_id` | string | arXiv identifier detected (e.g. `1610.05492`). |

## Heuristic seed fields (set by the parser, accepted or overridden by the reviewer)

| Column | Type | Meaning |
| --- | --- | --- |
| `publication_type_seed` | enum or blank | Seed value from the patterns in `topic_vocabulary.yaml:publication_type_patterns`. Blank when no pattern matched. |
| `pet_family_seed` | string | Semicolon-separated PET family labels detected by keyword. Multi-label. |
| `text_strategy` | enum | One of `abstract`, `summary`, `full`, `manual`. Determines what `acquire_text.py` does for this row. |

## Curation fields (set by the reviewer)

| Column | Type | Meaning |
| --- | --- | --- |
| `publication_type` | enum | The final publication type. Defaults to the seed value; edit to override. |
| `pet_family` | string | The final PET-family multi-label. Defaults to the seed value; edit to override. |
| `primary_topic` | enum | The final primary topic from the controlled vocabulary. Blank means "use the machine suggestion from `topic_suggestions.csv`". |
| `secondary_topics` | string | Optional, semicolon-separated. Used only where a source is substantively discussed under more than one survey section. |
| `review_note` | string | Free-text reviewer comment. Required whenever a seed is overridden or a low-margin assignment is confirmed against the suggestion. |

## Enum domains

`publication_type`:

```
academic_paper, survey_review, preprint, regulatory_guidance,
technical_whitepaper, standards_specification, industry_blog,
software_repository
```

`primary_topic`:

```
mechanism, systems, evaluation, assurance, governance, deployment, survey
```

`pet_family` (multi-label, `;`-separated):

```
differential_privacy, mpc, fhe, federated_learning, tee, zkp,
synthetic_data, syntactic_anonymisation
```

`text_strategy`:

```
abstract, summary, full, manual
```

## `data/source_text.csv` (Stage 2 output)

One row per source, recording the acquired text used for clustering and full
provenance. Regenerate with `python -m src.acquire_text`.

| Column | Meaning |
| --- | --- |
| `id` | Source id (joins to `sources.csv`). |
| `number` | Bibliography number, for convenience. |
| `text_strategy` | Acquisition strategy applied (`abstract` / `summary` / `full` / `manual`), derived from publication type. |
| `source` | Where the text came from: `arxiv`, `crossref`, `semantic_scholar`, `semantic_scholar_title`, `openalex`, `pdf`, `web_page`, `override`, `pdf_no_parser`, or `title_fallback`. |
| `status` | `ok` (text acquired) or `fallback` (no abstract/description obtainable; the build uses the title). |
| `http_status` | HTTP status of the last fetch attempt (blank for overrides). |
| `n_chars` | Length of the acquired text. |
| `title_seen` | The record/page title returned by the source. Used to check the abstract belongs to the right work; a mismatch with `sources.csv:title` flags a likely wrong DOI. |
| `text` | The acquired abstract / description (capped). Empty for fallbacks. |
