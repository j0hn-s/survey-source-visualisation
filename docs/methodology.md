# Methodology

This document is the canonical, long-form description of how the figure is
produced. The README is a summary; this is what a reviewer should read to
contest or extend any choice we have made.

The pipeline has seven explicit stages. Each stage produces a named artefact
on disk so that any single stage can be re-run without invalidating the
others.

```
reference_list.txt
        │  Stage 1
        ▼
data/sources.csv ────► (curated columns: publication_type, primary_topic, …)
        │  Stage 2
        ▼
data/source_text.csv    + data/abstracts_cache/{id}.txt overrides
        │  Stage 3
        ▼
text_normalised (in-memory column)
        │  Stage 4
        ▼
TF-IDF matrix + (optional) sentence embeddings
        │  Stage 5
        ▼
data/topic_suggestions.csv  (machine) →  reviewer  →  sources.csv:primary_topic
        │  Stage 6
        ▼
networkx.MultiGraph (edges typed topic | family | semantic)
        │  Stage 7
        ▼
figures/semantic_map_main.png + supporting figures
```

## Stage 1 — Source inventory and classification

The bibliography is the audited input. The parser
([src/parse_references.py](../src/parse_references.py)) extracts the fields
that can be read off the entry deterministically: id, number, raw text,
authors, year, title, venue, URL, DOI, and arXiv id. It also writes two
heuristic *seed* columns:

- `publication_type_seed` — derived from the regex patterns in
  `data/topic_vocabulary.yaml`. The patterns are deliberately conservative;
  any entry whose type cannot be inferred is left blank and flagged for
  manual review.
- `pet_family_seed` — multi-label, populated by substring match against the
  PET-family keyword sets.

**Why a seed rather than a final value?** Because publication type drives
text acquisition (Stage 2), getting it wrong silently corrupts everything
downstream. The seed/curation split makes the responsibility for the final
value explicit: a human reviewer either accepts the seed or overrides it,
with the override recorded in `review_note`.

### Publication-type taxonomy

| Label | What it means |
| --- | --- |
| `academic_paper` | Peer-reviewed conference or journal paper |
| `survey_review` | Survey, SoK, scoping review, Foundations & Trends monograph |
| `preprint` | arXiv, IACR ePrint, EasyChair, etc., not yet venue-published |
| `regulatory_guidance` | Regulator-issued policy, guidance, or framework |
| `technical_whitepaper` | Vendor or consortium technical white paper, solution brief, technical report |
| `standards_specification` | Formal standard or specification (ISO/IEC, W3C, EU regulation) |
| `industry_blog` | Practitioner blog post or short article |
| `software_repository` | Released code or library |

## Stage 2 — Text acquisition

This is the most methodologically loaded stage. Treating every source as a
single string and dropping it into TF-IDF would let a 200-page guideline
outweigh a four-page conference paper purely by length. We therefore apply a
**length-aware** acquisition rule keyed on publication type:

| Strategy | Applies to | Why |
| --- | --- | --- |
| `abstract` | academic_paper, survey_review, preprint | Abstracts are author-curated semantic summaries; they exist in standard, machine-readable form on CrossRef / arXiv. |
| `summary` | regulatory_guidance, technical_whitepaper, standards_specification | These are long, narrative documents; the executive summary or foreword is the authoritative statement of intent. Where no explicit summary exists we take the first ~1,500 words. |
| `full` | industry_blog | Blog posts are already short and scoped. |
| `manual` | software_repository, anything with an unknown type | Requires a human to supply a chosen excerpt. |

Concretely, [src/acquire_text.py](../src/acquire_text.py):

1. Looks for a human-supplied override at
   `data/abstracts_cache/{id}.txt` first. If present, it is used verbatim and
   recorded as `source = override`.
2. For `abstract` sources: queries CrossRef by DOI, then arXiv by id. If
   neither yields a payload the row is left blank and surfaced in the
   curation report.
3. For `summary` sources: reads a local PDF at
   `data/abstracts_cache/{id}.pdf` (the user supplies these) and extracts the
   first ~1,500 words with `pdfplumber`.
4. For `full` sources: requires a human-supplied `.txt` payload. We do not
   scrape arbitrary websites because the legality, encoding, and stability of
   doing so vary by publisher.

The set of sources for which acquisition was successful is reported at the
top of the notebook, so the reader can see exactly how much of the figure is
based on acquired text versus the title+venue fallback.

## Stage 3 — Normalisation

Implemented in [src/preprocess.py](../src/preprocess.py). The transformations
are the standard recipe used in classical bibliometric pipelines (Salton &
McGill, 1983) and in the text-mining module of VOSviewer (van Eck & Waltman,
2010):

1. lower-case;
2. strip URLs, DOIs, page ranges, numbers, and bibliographic noise;
3. replace curated multi-word PET phrases with single tokens
   (`differential privacy` → `differential_privacy`). The phrase list is
   read from `data/topic_vocabulary.yaml`, so changes to the controlled
   vocabulary automatically propagate. This step is borrowed from Mikolov et
   al. (2013, §4) but applied to a curated phrase list rather than a
   frequency threshold;
4. tokenise on whitespace and punctuation;
5. drop English stop words;
6. lemmatise.

## Stage 4 — Features

We compute two representations and use them for different jobs:

- **TF-IDF** ([Salton & Buckley, 1988](#references)) — sparse, interpretable.
  Used for topic scoring (Stage 5) and for the semantic-edge similarity
  matrix.
- **Sentence-Transformer embeddings** (`all-MiniLM-L6-v2`, Reimers &
  Gurevych, 2019) — dense, semantic. Used only as an *optional* alternative
  similarity source for the semantic edge type. The rest of the pipeline runs
  without them.

The defaults in [src/features.py](../src/features.py) drop hapax legomena
(`min_df=2`) and overly common terms (`max_df=0.8`), include bigrams, and
L2-normalise. These are deliberate, documented choices; they are not the
scikit-learn defaults.

## Stage 5 — Topic assignment

Implemented in
[src/topic_assignment.py](../src/topic_assignment.py).

A **closed** controlled vocabulary is defined in
`data/topic_vocabulary.yaml`. It deliberately mirrors the structure of the
survey paper:

| Topic | Substantive meaning |
| --- | --- |
| `mechanism` | Original cryptographic / statistical primitive |
| `systems` | Engineering — compiler, library, SDK, hardware acceleration |
| `evaluation` | Benchmark, performance, comparative study |
| `assurance` | Threat model, attack, audit, registry, verification |
| `governance` | Regulation, policy, standard, ethics, framework |
| `deployment` | Sectoral application or real-world case study |
| `survey` | Survey, SoK, review, Foundations & Trends monograph |

Why not unsupervised topic modelling (LDA, Blei et al. 2003)? Because the
purpose of the figure is to show how sources cluster **relative to the
structure of our paper**. An unsupervised topic model would discover its own
topics, which need not match the paper at all — defeating the point of
inserting the figure in the first place.

### Scoring procedure

1. Build a pseudo-document per topic (the concatenation of its keywords).
2. Fit a single TF-IDF vectoriser over the union of (topic pseudo-documents,
   source texts).
3. Compute cosine similarity between every source and every topic in this
   shared space.
4. Rank topics per source. Record the primary topic, the runner-up, the
   scores, and the margin (primary − runner-up).
5. Flag any source whose margin is below `min_margin` (default 0.05) for
   mandatory human review. A reviewer is also expected to spot-check
   confident assignments; spot-check size is documented in the curation
   guide.

The output is `data/topic_suggestions.csv`. The reviewer reads it alongside
the source itself and decides the final value of `sources.csv:primary_topic`.
The notebook merges suggestions and curated values, always preferring the
human value.

PET-family tags are independent multi-labels. Each family is detected by
keyword presence in the text payload; this is conservative on purpose
(presence ≥ 1, not weighted scoring) because PET-family attribution rarely
needs the precision that primary-topic attribution does.

## Stage 6 — Graph construction

[src/graph.py](../src/graph.py) builds a `networkx.MultiGraph` with three
edge types. Each edge carries `etype ∈ {topic, family, semantic}`.

| Edge type | Rule | Borrowed from |
| --- | --- | --- |
| `topic` | Two sources share `primary_topic` | Standard cluster-as-edges encoding |
| `family` | Two sources share at least one PET family | Bibliographic-coupling analogue (Kessler, 1963) — co-membership in a topical class implies relatedness |
| `semantic` | Cosine similarity above a threshold, capped at k nearest neighbours | k-NN association cap from VOSviewer (van Eck & Waltman, 2010, §3.2) |

The k-NN cap exists because a dense similarity graph collapses into a
"hairball" that hides structure. The threshold (default 0.20) is chosen to
keep about 1–2 semantic neighbours per node on average; both `k` and
`threshold` are exposed as notebook parameters and their effect on density
is reported.

## Stage 7 — Rendering

[src/visualise.py](../src/visualise.py).

- **Layout** — Fruchterman & Reingold (1991), the de-facto default for
  bibliometric network figures since CiteSpace (Chen, 2006) and VOSviewer
  popularised it. We seed the random state for reproducibility.
- **Encoding** — colour for primary topic; shape for source type; size for
  degree. We avoid encoding more than three dimensions on a single mark
  (Bertin, 1967).
- **Colour palette** — colour-blind-safe qualitative scheme drawn from
  ColorBrewer (Brewer, 1994). Topic colours are stable across renderings.
- **Supporting figures**:
  - timeline of sources by year and topic;
  - publication-type × primary-topic heatmap (with explicit axis labels and
    a colour-bar legend so the matrix is interpretable in isolation);
  - PET-family breakdown — horizontal stacked bar over the seven PET families
    the survey explicitly considers (secure MPC, homomorphic encryption,
    differential privacy, synthetic data, zero-knowledge, federated learning
    and distributed analytics, trusted execution environments). The bars are
    multi-label: a source tagged `differential_privacy;federated_learning`
    contributes one unit to both. The total across bars therefore exceeds the
    number of sources, by design. Sources that carry no tag from the seven
    are summarised in a separate "Other / none" bar; they are not silently
    dropped. `syntactic_anonymisation` is tracked in metadata but does not
    appear as a bar - the figure answers "which of the seven survey-defined
    PETs do the sources focus on?", not "which anonymisation technique do
    they use?".
- **Interactive HTML** — produced by `pyvis` for the appendix or repository
  reader who wants to hover for tooltips.

## Limitations and what to read into the figure

- The figure depicts *the sources considered in this survey*, not the PETs
  literature.
- Topic assignment is guided by a closed taxonomy. A different team with
  different topics would produce a visibly different figure from identical
  inputs.
- Text payloads are deliberately length-controlled. A short policy summary
  and a long technical paper are treated as comparable inputs; this avoids
  page-count dominance but flattens substantive depth.
- The figure should be read as evidence about *our reading list*, not as a
  ground-truth map of an evolving field. Patterns it surfaces (e.g.
  governance over-representation) are findings to comment on, not flaws.

## References

- Bertin, J. (1967) *Sémiologie graphique*. Paris: Mouton.
- Blei, D.M., Ng, A.Y. and Jordan, M.I. (2003) ‘Latent Dirichlet allocation’, *Journal of Machine Learning Research*, 3, pp. 993–1022.
- Brewer, C.A. (1994) ‘Color use guidelines for mapping and visualization’, in MacEachren, A.M. and Taylor, D.R.F. (eds.) *Visualization in Modern Cartography*. Oxford: Pergamon, pp. 123–147.
- Chen, C. (2006) ‘CiteSpace II: Detecting and visualizing emerging trends and transient patterns in scientific literature’, *Journal of the American Society for Information Science and Technology*, 57(3), pp. 359–377.
- Fruchterman, T.M.J. and Reingold, E.M. (1991) ‘Graph drawing by force-directed placement’, *Software: Practice and Experience*, 21(11), pp. 1129–1164.
- Kessler, M.M. (1963) ‘Bibliographic coupling between scientific papers’, *American Documentation*, 14(1), pp. 10–25.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S. and Dean, J. (2013) ‘Distributed representations of words and phrases and their compositionality’, in *Advances in Neural Information Processing Systems 26*.
- Page, M.J. et al. (2021) ‘The PRISMA 2020 statement: An updated guideline for reporting systematic reviews’, *BMJ*, 372, n71.
- Reimers, N. and Gurevych, I. (2019) ‘Sentence-BERT: Sentence embeddings using Siamese BERT-networks’, *EMNLP 2019*.
- Rethlefsen, M.L. et al. (2021) ‘PRISMA-S: An extension to the PRISMA statement for reporting literature searches in systematic reviews’, *Systematic Reviews*, 10(1), 39.
- Salton, G. and Buckley, C. (1988) ‘Term-weighting approaches in automatic text retrieval’, *Information Processing & Management*, 24(5), pp. 513–523.
- Salton, G. and McGill, M.J. (1983) *Introduction to Modern Information Retrieval*. New York: McGraw-Hill.
- Small, H. (1973) ‘Co-citation in the scientific literature: A new measure of the relationship between two documents’, *Journal of the American Society for Information Science*, 24(4), pp. 265–269.
- van Eck, N.J. and Waltman, L. (2010) ‘Software survey: VOSviewer, a computer program for bibliometric mapping’, *Scientometrics*, 84(2), pp. 523–538.
