"""Stage 3 - normalise text into a comparable form.

The transformations applied here mirror those used in classical bibliometric
analysis pipelines (e.g. Salton & McGill, 1983; van Eck & Waltman, 2010, who
follow the same lower-case-lemmatise-stop-word recipe in the VOSviewer
text-mining module). Their purpose is to make different source types
comparable on equal footing once their length has already been controlled by
the acquisition stage.

The transformations are:

    1. lower-case
    2. strip URLs, in-text citation markers, equation residues, and
       bibliography artefacts (page ranges, DOI strings)
    3. replace selected multi-word PET phrases with single tokens
       ("differential privacy" -> "differential_privacy") so that TF-IDF
       sees them as atomic. This step is borrowed from Mikolov et al.
       (2013, §4) where common bigrams are merged before training, applied
       here to a curated PET phrase list rather than a frequency threshold.
    4. tokenise on whitespace and punctuation
    5. drop English stop words
    6. lemmatise verbs and nouns

The phrase list is read from data/topic_vocabulary.yaml so that changes to
the controlled vocabulary automatically propagate.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import yaml

# Lazy import so that an environment without NLTK can still parse_references.
_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk  # type: ignore

    for resource in ("punkt_tab", "stopwords", "wordnet"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK_READY = True


def _collect_phrases(vocab_path: Path) -> list[str]:
    """Multi-word phrases from the topic vocabulary that we want to keep atomic."""
    vocab = yaml.safe_load(vocab_path.read_text(encoding="utf-8"))
    phrases: set[str] = set()
    for group in ("topics", "pet_families"):
        for entry in vocab.get(group, []):
            for kw in entry.get("keywords", []):
                if " " in kw or "-" in kw:
                    phrases.add(kw.lower())
    # Longest first so that "differential privacy" is replaced before "privacy".
    return sorted(phrases, key=len, reverse=True)


URL_RE = re.compile(r"https?://\S+|<[^>]+>")
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+")
PAGE_RE = re.compile(r"\bpp?\.\s*\d+(?:[-–]\d+)?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")


def _join_phrase(phrase: str) -> str:
    """Convert "differential privacy" -> "differential_privacy"."""
    return re.sub(r"[\s\-]+", "_", phrase.strip())


def normalise(
    text: str,
    phrases: list[str],
    stopwords: set[str],
    lemmatiser,
) -> str:
    if not text:
        return ""
    out = text.lower()
    out = URL_RE.sub(" ", out)
    out = DOI_RE.sub(" ", out)
    out = PAGE_RE.sub(" ", out)
    for phrase in phrases:
        joined = _join_phrase(phrase)
        out = re.sub(re.escape(phrase), joined, out)
    out = NUMBER_RE.sub(" ", out)
    out = PUNCT_RE.sub(" ", out)

    tokens = []
    for tok in out.split():
        if tok in stopwords or len(tok) < 3:
            continue
        tokens.append(lemmatiser.lemmatize(tok))
    return " ".join(tokens)


def normalise_series(texts, vocab_path: Path):
    """Normalise an iterable of strings.

    Returns a list of strings of the same length. Empty inputs map to "".
    """
    _ensure_nltk()
    from nltk.corpus import stopwords as _sw  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore

    phrases = _collect_phrases(vocab_path)
    stop = set(_sw.words("english"))
    lemma = WordNetLemmatizer()
    return [normalise(t or "", phrases, stop, lemma) for t in texts]
