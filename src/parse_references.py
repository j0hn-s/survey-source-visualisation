"""Stage 1 - parse the Harvard-style reference list into structured metadata.

The input is a numbered, hand-curated list of references (one per line) using
Cite Them Right Harvard. We extract only fields that can be read off the
reference itself without ambiguity:

    * id            stable, content-derived identifier (ref-XXXXXXXX)
    * number        the position in the bibliography as given by the author
    * raw           the verbatim entry
    * authors       text before the first '(YYYY' token
    * year          first 4-digit year inside parentheses
    * title         text between the first pair of typographic quotes
    * venue         text after the title up to the URL or end of line
    * url           the first <...> URL if present
    * publication_type_seed   best-effort heuristic; see topic_vocabulary.yaml
    * pet_family_seed         best-effort multi-label heuristic over the raw entry

The two *_seed columns are deliberately named "seed" because they are starting
points for human review, not authoritative labels. See
docs/topic_assignment_guide.md for the curation workflow.

Stable id design
----------------
`id` is `ref-<first 8 hex chars of SHA-1 of the normalised raw entry>`. This
matters when the bibliography is later extended:

  * inserting a reference in the middle of `reference_list.txt` no longer
    shifts every downstream id;
  * curated rows in `sources.csv` survive a re-parse because they are looked
    up by id, not position;
  * cached text payloads at `data/abstracts_cache/{id}.txt` remain valid.

If you edit an existing entry's text (e.g. fix a typo in the bibliography),
its hash changes and the parser will treat it as a new entry. The merge
report flags the old id as removed and the new id as added so you can choose
to migrate any curated columns explicitly.

Re-running the parser
---------------------
By default the parser runs in **merge mode**: it reads any existing
`data/sources.csv`, indexes it by id, and copies the curated columns
(`publication_type`, `pet_family`, `primary_topic`, `secondary_topics`,
`review_note`) onto the freshly parsed rows. New entries appear with empty
curated columns; entries no longer present in `reference_list.txt` are
dropped, and a list of dropped ids is printed to stderr.

Run:

    python -m src.parse_references                       # merge (default)
    python -m src.parse_references --rebuild             # discard curation
    python -m src.parse_references --input X --output Y

Methodological note: we extract from the bibliography rather than re-querying
external systems at this stage so that the seed CSV is reproducible from a
single committed input. Abstract acquisition is a later, explicit stage
(see src/acquire_text.py) and writes to a separate cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import yaml


# Columns the parser overwrites on every run (mechanical / seed fields).
MECHANICAL_COLUMNS: tuple[str, ...] = (
    "id", "number", "raw", "authors", "year", "title", "venue",
    "url", "doi", "arxiv_id",
    "publication_type_seed", "pet_family_seed", "text_strategy",
)

# Columns the parser preserves across re-parses (human-curation fields). On a
# merge, these are copied from the existing sources.csv by id.
CURATED_COLUMNS: tuple[str, ...] = (
    "publication_type", "pet_family", "primary_topic",
    "secondary_topics", "review_note",
)


# Single typographic and ASCII quote families used in the source file.
OPEN_QUOTES = "‘’“”'\""
CLOSE_QUOTES = "‘’“”'\""

ENTRY_RE = re.compile(r"^\s*(\d+)\.\s*(.+)$")
YEAR_RE = re.compile(r"\((\d{4})[a-z]?(?:/\d{4})?\)")
URL_RE = re.compile(r"<\s*(https?://[^>\s]+)\s*>")
DOI_RE = re.compile(r"https?://doi\.org/([^\s<>]+)", re.IGNORECASE)
ARXIV_RE = re.compile(r"arXiv[:\s]*([0-9]{4}\.[0-9]{4,5})", re.IGNORECASE)


@dataclass
class Source:
    id: str
    number: int
    raw: str
    authors: str = ""
    year: int | None = None
    title: str = ""
    venue: str = ""
    url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    publication_type_seed: str = ""
    pet_family_seed: str = ""
    # Reserved for human curation; left blank by the parser.
    publication_type: str = ""
    pet_family: str = ""
    primary_topic: str = ""
    secondary_topics: str = ""
    text_strategy: str = ""
    review_note: str = ""

    def as_row(self) -> dict[str, str]:
        d = asdict(self)
        d["year"] = "" if self.year is None else str(self.year)
        return d


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" ,.;:")


def _content_id(body: str) -> str:
    """Stable, content-derived id for an entry.

    The hash is taken over the *normalised* raw text (whitespace collapsed,
    leading "N." numbering stripped) so that:
      * re-numbering the bibliography does not change the id;
      * adding or removing whitespace does not change the id;
      * fixing a substantive typo *does* change the id (the merge report
        will surface this as one removed and one added id, prompting an
        explicit migration decision).
    """
    normalised = re.sub(r"\s+", " ", body.strip())
    digest = hashlib.sha1(normalised.encode("utf-8")).hexdigest()
    return f"ref-{digest[:8]}"


def _extract_title(body: str) -> tuple[str, str]:
    """Return (title, remainder). The title is the first balanced quoted span.

    Harvard entries use typographic quotes; we accept ASCII as a fallback. We
    deliberately do not try to handle nested quotes; if extraction fails we
    return an empty title and the original body so the row remains auditable.
    """
    open_pos = -1
    for i, ch in enumerate(body):
        if ch in OPEN_QUOTES:
            open_pos = i
            break
    if open_pos == -1:
        return "", body
    close_pos = -1
    for j in range(open_pos + 1, len(body)):
        if body[j] in CLOSE_QUOTES and body[j] != body[open_pos]:
            close_pos = j
            break
        if body[j] == body[open_pos] and body[j] in "'\"":
            close_pos = j
            break
    if close_pos == -1:
        return "", body
    title = body[open_pos + 1 : close_pos]
    remainder = body[:open_pos] + body[close_pos + 1 :]
    return _strip(title), remainder


def _classify_publication_type(raw: str, patterns: list[dict]) -> str:
    for spec in patterns:
        for pat in spec["patterns"]:
            if pat in raw:
                return spec["label"]
    return ""


def _classify_pet_families(raw: str, families: list[dict]) -> list[str]:
    lowered = raw.lower()
    hits: list[str] = []
    for spec in families:
        for kw in spec["keywords"]:
            if kw.lower() in lowered:
                hits.append(spec["label"])
                break
    return hits


def _infer_text_strategy(publication_type: str) -> str:
    """Map publication type to the text-acquisition rule from methodology §2.

    Strategies:
      abstract        - retrieve and analyse the abstract only
      summary         - retrieve and analyse the executive summary / first
                        ~1,500 words if no explicit summary exists
      full            - the source is short enough to use in full
      manual          - the parser could not decide; a human must specify
    """
    mapping = {
        "academic_paper": "abstract",
        "survey_review": "abstract",
        "preprint": "abstract",
        "regulatory_guidance": "summary",
        "technical_whitepaper": "summary",
        "standards_specification": "summary",
        "industry_blog": "full",
        "software_repository": "manual",
    }
    return mapping.get(publication_type, "manual")


def parse_entry(line: str, vocab: dict) -> Source | None:
    m = ENTRY_RE.match(line)
    if not m:
        return None
    number = int(m.group(1))
    body = m.group(2).strip()

    year_match = YEAR_RE.search(body)
    year = int(year_match.group(1)) if year_match else None
    authors = _strip(body[: year_match.start()]) if year_match else ""

    after_year = body[year_match.end():] if year_match else body
    title, venue = _extract_title(after_year)
    venue = _strip(venue)

    url_match = URL_RE.search(body)
    url = url_match.group(1) if url_match else ""
    doi_match = DOI_RE.search(body)
    doi = doi_match.group(1) if doi_match else ""
    arxiv_match = ARXIV_RE.search(body)
    arxiv_id = arxiv_match.group(1) if arxiv_match else ""

    pub_type = _classify_publication_type(body, vocab["publication_type_patterns"])
    pet_families = _classify_pet_families(body, vocab["pet_families"])
    text_strategy = _infer_text_strategy(pub_type)

    return Source(
        id=_content_id(body),
        number=number,
        raw=body,
        authors=authors,
        year=year,
        title=title,
        venue=venue,
        url=url,
        doi=doi,
        arxiv_id=arxiv_id,
        publication_type_seed=pub_type,
        pet_family_seed=";".join(pet_families),
        publication_type=pub_type,
        pet_family=";".join(pet_families),
        text_strategy=text_strategy,
    )


def parse_file(path: Path, vocab: dict) -> list[Source]:
    out: list[Source] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src = parse_entry(line, vocab)
        if src is not None:
            out.append(src)
    return out


def write_csv(sources: Iterable[Source], path: Path) -> None:
    rows = [s.as_row() for s in sources]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class MergeReport:
    """Summary of a re-parse against an existing sources.csv.

    Read this to find out what changed: which entries are new, which were
    dropped (i.e. removed from the bibliography or had their text edited),
    and how many curated rows survived the merge.
    """
    total_now: int
    total_before: int
    added_ids: list[str] = field(default_factory=list)
    removed_ids: list[str] = field(default_factory=list)
    curated_preserved: int = 0
    curated_dropped: int = 0
    duplicates: list[tuple[str, list[int]]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"parsed {self.total_now} entries (previously {self.total_before})",
            f"  added:   {len(self.added_ids)}",
            f"  removed: {len(self.removed_ids)}",
            f"  curated rows preserved: {self.curated_preserved}",
        ]
        if self.curated_dropped:
            lines.append(
                f"  curated rows DROPPED:  {self.curated_dropped}  "
                "(entries removed from reference_list.txt or had text edited; "
                "see removed_ids)"
            )
        if self.duplicates:
            lines.append(
                f"  duplicate entries collapsed: {len(self.duplicates)}  "
                "(same content under multiple bibliography numbers)"
            )
            for sid, numbers in self.duplicates[:5]:
                lines.append(f"    {sid}: numbers {numbers}")
        if self.removed_ids:
            shown = ", ".join(self.removed_ids[:5])
            more = "" if len(self.removed_ids) <= 5 else f"  (+{len(self.removed_ids) - 5} more)"
            lines.append(f"  removed_ids: {shown}{more}")
        if self.added_ids:
            shown = ", ".join(self.added_ids[:5])
            more = "" if len(self.added_ids) <= 5 else f"  (+{len(self.added_ids) - 5} more)"
            lines.append(f"  added_ids:   {shown}{more}")
        return "\n".join(lines)


def _read_existing(path: Path) -> dict[str, dict[str, str]]:
    """Index an existing sources.csv by id. Missing file returns an empty dict."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["id"]: row for row in reader if row.get("id")}


def _has_curation(row: dict[str, str]) -> bool:
    """A row counts as curated only when a reviewer has added value beyond
    the parser's seed values.

    `publication_type` and `pet_family` are populated by the parser with the
    seed value on first parse, so a non-empty value alone does not imply a
    human decision; it must *differ* from the seed. `primary_topic`,
    `secondary_topics`, and `review_note` are blank by default, so any
    non-empty value is human-authored.
    """
    if row.get("primary_topic", "").strip():
        return True
    if row.get("secondary_topics", "").strip():
        return True
    if row.get("review_note", "").strip():
        return True
    pt = row.get("publication_type", "").strip()
    pt_seed = row.get("publication_type_seed", "").strip()
    if pt and pt != pt_seed:
        return True
    fam = row.get("pet_family", "").strip()
    fam_seed = row.get("pet_family_seed", "").strip()
    if fam and fam != fam_seed:
        return True
    return False


def merge_sources(
    parsed: list[Source],
    existing: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], MergeReport]:
    """Combine freshly parsed entries with curated columns from a prior run.

    For every unique parsed id we keep the mechanical / seed fields from the
    parser (they reflect the current text) and overlay any non-empty curated
    columns from the prior run keyed on `id`. New entries appear with the
    parser's seed values intact; entries that were curated but are no longer
    present in `reference_list.txt` are dropped and reported.

    Two entries with identical normalised text collapse to one row. The
    canonical row keeps the lowest bibliography `number` (the earliest
    occurrence). All collapsed numbers are reported in the `duplicates`
    field of the MergeReport so the bibliography author can decide whether
    to remove the duplicates from `reference_list.txt`.
    """
    # Group parsed entries by stable id and capture every position number.
    grouped: dict[str, list[Source]] = {}
    for s in parsed:
        grouped.setdefault(s.id, []).append(s)

    duplicates: list[tuple[str, list[int]]] = []
    canonical: list[Source] = []
    for sid, group in grouped.items():
        group_sorted = sorted(group, key=lambda s: s.number)
        canonical.append(group_sorted[0])
        if len(group_sorted) > 1:
            duplicates.append((sid, [s.number for s in group_sorted]))

    parsed_by_id = {s.id: s for s in canonical}
    added = [sid for sid in parsed_by_id if sid not in existing]
    removed = [sid for sid in existing if sid not in parsed_by_id]
    curated_preserved = 0

    merged_rows: list[dict[str, str]] = []
    for source in sorted(canonical, key=lambda s: s.number):
        row = source.as_row()
        prior = existing.get(source.id)
        if prior:
            had_curation = _has_curation(prior)
            for col in CURATED_COLUMNS:
                value = prior.get(col, "")
                if value:
                    row[col] = value
            if had_curation:
                curated_preserved += 1
        merged_rows.append(row)

    curated_dropped = sum(
        1 for sid in removed if _has_curation(existing[sid])
    )

    report = MergeReport(
        total_now=len(canonical),
        total_before=len(existing),
        added_ids=added,
        removed_ids=removed,
        curated_preserved=curated_preserved,
        curated_dropped=curated_dropped,
        duplicates=duplicates,
    )
    return merged_rows, report


def _write_rows(rows: list[dict[str, str]], path: Path) -> None:
    """Write a merge result. The column order is fixed (mechanical then curated)
    so the CSV diffs cleanly across re-runs."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(MECHANICAL_COLUMNS) + list(CURATED_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="reference_list.txt")
    ap.add_argument("--output", default="data/sources.csv")
    ap.add_argument("--vocab", default="data/topic_vocabulary.yaml")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "discard any existing sources.csv and start fresh. Curated columns "
            "are NOT preserved. Use only when migrating id schemes or starting over."
        ),
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    vocab = yaml.safe_load((root / args.vocab).read_text(encoding="utf-8"))
    parsed = parse_file(root / args.input, vocab)

    out_path = root / args.output
    if args.rebuild:
        existing: dict[str, dict[str, str]] = {}
    else:
        existing = _read_existing(out_path)

    rows, report = merge_sources(parsed, existing)
    _write_rows(rows, out_path)
    print(report.render())
    print(f"wrote -> {args.output}")
    if report.curated_dropped:
        sys.exit(2)  # non-zero so a script can surface lost curation


if __name__ == "__main__":
    main()
