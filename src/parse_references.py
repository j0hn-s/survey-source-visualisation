"""Stage 1 - parse the Harvard-style reference list into structured metadata.

The input is a numbered, hand-curated list of references (one per line) using
Cite Them Right Harvard. We extract only fields that can be read off the
reference itself without ambiguity:

    * id            stable identifier (refNNN)
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

Run:

    python -m src.parse_references \\
        --input reference_list.txt --output data/sources.csv

Methodological note: we extract from the bibliography rather than re-querying
external systems at this stage so that the seed CSV is reproducible from a
single committed input. Abstract acquisition is a later, explicit stage
(see src/acquire_text.py) and writes to a separate cache.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import yaml


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
        id=f"ref{number:03d}",
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="reference_list.txt")
    ap.add_argument("--output", default="data/sources.csv")
    ap.add_argument("--vocab", default="data/topic_vocabulary.yaml")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    vocab = yaml.safe_load((root / args.vocab).read_text(encoding="utf-8"))
    sources = parse_file(root / args.input, vocab)
    write_csv(sources, root / args.output)
    print(f"parsed {len(sources)} entries -> {args.output}")


if __name__ == "__main__":
    main()
