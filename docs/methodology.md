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
        ├──► data/semantic_clusters.csv   (emergent clusters - DRIVES figure colour)
        └──► data/topic_suggestions.csv   (controlled-vocab scoring - human-review aid)
                                              │  reviewer  →  sources.csv:primary_topic
        │  Stage 6
        ▼
networkx.MultiGraph (edges typed family | semantic; coloured by cluster)
        │  Stage 7
        ▼
figures/semantic_map_main.png + supporting figures + data/figure_metrics.json
```

## Stage 1 — Source inventory and classification

The bibliography is the audited input. The parser
([src/parse_references.py](../src/parse_references.py)) extracts the fields
that can be read off the entry deterministically: id, number, raw text,
authors, year, title, venue, URL, DOI, and arXiv id.

### Stable, content-derived ids

`id` is `ref-<first 8 hex chars of SHA-1 over the normalised raw entry>`. The
hash is taken over the entry text after whitespace is collapsed and the
leading bibliography number is stripped. The consequences:

- inserting a reference in the middle of `reference_list.txt` does **not**
  shift the ids of any other entry;
- curated rows in `sources.csv` survive a re-parse because they are looked
  up by id, not position;
- cached text payloads at `data/abstracts_cache/{id}.txt` remain valid
  across re-parses;
- two bibliography entries with identical normalised text collapse to one
  row (the parser keeps the lowest position number and reports the others
  as duplicates).

A side-effect that the methodology takes as a feature rather than a bug:
fixing a substantive typo changes the id. The merge report then shows one
removed id and one added id, prompting an explicit migration decision rather
than silently re-attaching curation to a different text.

### Merge mode

A re-parse against an existing `sources.csv` preserves the reviewer-owned
columns (`primary_topic`, `secondary_topics`, `review_note`) and any
publication-type or PET-family value that differs from the parser's seed.
Rows are matched on id. Added, removed, and collapsed-duplicate ids are
reported. The parser's exit code is non-zero when previously curated rows
are dropped, so the workflow can be wrapped in a script that surfaces lost
curation.

It also writes two heuristic *seed* columns:

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

This is the most methodologically loaded stage: the clusters are built from
this text, so **what we feed in per source is the core modelling decision.**
Treating every source as one string would let a 200-page guideline outweigh a
four-page paper by length alone, and different source types carry their meaning
in different places. We therefore acquire a **type-appropriate,
length-controlled** payload, keyed on the `text_strategy` that
`parse_references.py` derives from publication type. Implemented in
[src/acquire_text.py](../src/acquire_text.py); provenance for every source is
written to `data/source_text.csv`.

### The per-type text model, and the assumption behind each

| Strategy | Source types | What we acquire | Modelling assumption |
| --- | --- | --- | --- |
| `abstract` | academic_paper, preprint, survey_review | The author-written abstract | The abstract is a faithful, lower-variance summary of the contribution. |
| `summary` | regulatory_guidance, technical_whitepaper, standards_specification | The document's own description / scope: page meta-description + opening paragraphs (≈first 1,500 chars) | For long narrative documents the opening states intent. |
| `full` | industry_blog | Page description + lead paragraphs | Blogs are already short and scoped. |
| `manual` | online_resource, software_repository | Landing-page / repository description (meta-description, README lead) | The "high-level description equivalent" for things with no abstract: a project's self-description states its purpose. |

Two assumptions span all types:

1. **Stated scope, not full content.** We map what a source says it is about,
   not its full text. This is deliberate (length control) and is why a one-line
   policy summary and a long technical paper enter on comparable footing.
2. **The title is always informative.** The text fed to the model is the title
   joined to the acquired body (see `build_figures._model_text`), capped at
   1,800 characters. A source therefore is never empty.

### Acquisition procedure and escalation

A human-supplied override at `data/abstracts_cache/{id}.txt` always wins. Then,
for `abstract` sources, the chain is **arXiv API → Crossref → Semantic Scholar
(by id) → OpenAlex (reconstructed from its inverted index) → landing page →
Semantic Scholar (title search, with a match guard)**. For `summary` / `full` /
`manual` sources the landing page is the description source (PDFs are parsed
with `pdfplumber`; a browser user-agent is used because many publisher and
government pages reject non-browser agents). Only public bibliographic APIs and
public landing pages are used, and **text is never invented**: when nothing is
obtainable the source falls back to its title and is recorded as
`source = title_fallback`.

### Coverage and integrity

`source_text.csv` records, for every source: the `text_strategy`, the `source`
it was acquired from, a `status` (`ok` / `fallback`), the `http_status`,
`n_chars`, and the `title_seen` returned by the API. This lets a reader confirm
that **every** abstract or equivalent was considered and see exactly where each
came from (currently ≈88% acquired, the remainder title fallbacks — mostly
recent papers with no open abstract and bare landing pages). Because the API
sources also return the record's title, a mismatch between that title and ours
flags a likely wrong DOI in the reference list rather than a clustering error;
[tests/test_source_text.py](../tests/test_source_text.py) surfaces these.

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

## Stage 5 — Theme detection

Stage 5 answers "what themes are in this reading list?" by two parallel
routes. **Stage 5a is what colours the figure.** Stage 5b is a complementary,
paper-aligned lens retained for the human-review workflow.

### Stage 5a — Emergent semantic clusters (drives the figure)

Implemented in [src/semantic_clusters.py](../src/semantic_clusters.py). This is
science mapping in the bibliometric sense: the clusters are *discovered* from
the corpus rather than imposed.

The similarity is computed over the **abstracts** acquired in Stage 2 (title +
abstract / description), not titles, so the structure reflects what each source
is actually about.

1. **Sparsify.** Build a k-nearest-neighbour graph over the TF-IDF cosine
   similarity matrix (k = 12, similarity ≥ 0.10). This is the association cap
   VOSviewer uses (van Eck & Waltman, 2010, §3.2) to stop a dense similarity
   matrix collapsing into a hairball.
2. **Normalise to association strength.** Re-weight each surviving edge to
   `a_ij = w_ij · 2m / (s_i · s_j)`, where `s_i` is the weighted degree of `i`
   and `2m` the total edge weight. This is van Eck & Waltman's normalisation:
   it corrects for a source that is similar to many others co-occurring
   strongly by chance.
3. **Partition by modularity.** Run the Louvain method (Blondel et al., 2008;
   modularity objective, Newman, 2006) at resolution 0.6 on the
   association-strength graph. The number of clusters is **not** chosen by us;
   it emerges from the structure (currently nine clusters plus a small "other"
   bucket of three sparsely-connected sources, with modularity ≈ 0.64).
4. **Label.** Give each cluster its most characteristic terms by class-based
   TF-IDF — concatenate every abstract in a cluster into one pseudo-document
   and rank terms by TF-IDF across clusters (with `max_df` dropping terms
   common to most clusters, such as "data"). This is the classical analogue of
   the c-TF-IDF labelling popularised by BERTopic (Grootendorst, 2022).

**Reproducibility.** The partition is a deterministic function of
`(similarity matrix, k, threshold, resolution, seed)`. Louvain's tie-breaking
iterates over sets of node labels whose order varies per process under Python
hash randomisation; we therefore relabel nodes to integers in sorted order
before partitioning, and break size ties by smallest member id, so the result
is identical across processes and hash seeds. This is what
[tests/test_semantic_clusters.py](../tests/test_semantic_clusters.py) pins down.

**Why emergent clustering, not the controlled vocabulary, for the colour?**
Colouring by the a-priori controlled-vocabulary topic (Stage 5b) scores each
source against fixed keyword bags, which produced near-tie margins and an
authoritative-looking near-coin-flip colour. The general objection to
unsupervised topic models (LDA, Blei et al. 2003) is that they fit their own
generative topics, unrelated to the figure. Louvain community detection avoids
that objection: it operates on the *same* abstract-similarity graph the map
already draws, so the clusters are exactly the structure a reader sees, and
they are labelled by real terms rather than latent dimensions. Reassuringly,
the emergent clusters recover the survey's own themes (federated learning,
confidential computing, health-data governance, synthetic data, secure
multi-party computation, homomorphic encryption, zero-knowledge proofs, PETs
policy, anonymity and record linkage) without being told them.

### Stage 5b — Controlled-vocabulary topic scoring (human-review aid)

Implemented in [src/topic_assignment.py](../src/topic_assignment.py). A
**closed** controlled vocabulary in `data/topic_vocabulary.yaml` mirrors the
structure of the survey paper:

| Topic | Substantive meaning |
| --- | --- |
| `mechanism` | Original cryptographic / statistical primitive |
| `systems` | Engineering — compiler, library, SDK, hardware acceleration |
| `evaluation` | Benchmark, performance, comparative study |
| `assurance` | Threat model, attack, audit, registry, verification |
| `governance` | Regulation, policy, standard, ethics, framework |
| `deployment` | Sectoral application or real-world case study |
| `survey` | Survey, SoK, review, Foundations & Trends monograph |

Each source is scored against every topic by cosine similarity in a shared
TF-IDF space (topic pseudo-documents ∪ source texts), recording the primary
topic, runner-up, and margin. Sources with margin below `min_margin` (0.05)
are flagged for mandatory human review. The output is
`data/topic_suggestions.csv`; a reviewer reads it alongside the source and sets
`sources.csv:primary_topic`. This lens is **not** the figure colour (see
above), but it is the right structure for a human deciding how a source maps
onto the paper's sections.

PET-family tags are independent multi-labels, detected by keyword presence in
the text payload (presence ≥ 1, not weighted scoring); they colour the
supporting PET-family figure and contribute the `family` edges in Stage 6.

## Stage 6 — Graph construction

[src/graph.py](../src/graph.py) builds a `networkx.MultiGraph` with three
edge types. Each edge carries `etype ∈ {topic, family, semantic}`.

| Edge type | Rule | Borrowed from |
| --- | --- | --- |
| `topic` | Two sources share `primary_topic` | Standard cluster-as-edges encoding (used only when a curated topic exists) |
| `family` | Two sources share at least one PET family | Bibliographic-coupling analogue (Kessler, 1963) — co-membership in a topical class implies relatedness |
| `semantic` | Cosine similarity above a threshold, capped at k nearest neighbours | k-NN association cap from VOSviewer (van Eck & Waltman, 2010, §3.2) |

The k-NN cap exists because a dense similarity graph collapses into a
"hairball" that hides structure. Both `k` and `threshold` are exposed as
parameters and their effect on density is reported.

The headless science-mapping build ([src/build_figures.py](../src/build_figures.py))
uses the **same** k-NN semantic graph for clustering and for the drawn edges
(k = 10, threshold = 0.12), so edges and clusters are consistent. It does
**not** add topic edges — the figure is coloured by emergent cluster (Stage
5a) and the layout is driven by semantic edges (with a faint PET-family pull),
so the clusters separate spatially rather than being forced apart by a topic
grouping. Topic edges are still produced by the notebook path when a curated
`primary_topic` column is present.

## Stage 7 — Rendering

[src/visualise.py](../src/visualise.py).

- **Layout** — Fruchterman & Reingold (1991), the de-facto default for
  bibliometric network figures since CiteSpace (Chen, 2006) and VOSviewer
  popularised it. We seed the random state for reproducibility.
- **Encoding** — colour for emergent semantic cluster (Stage 5a); shape for
  source type; size for degree. We avoid encoding more than three dimensions
  on a single mark (Bertin, 1967).
- **Colour palette** — colour-blind-safe qualitative scheme drawn from
  ColorBrewer (Brewer, 1994). Cluster colours are stable across renderings
  because the partition is deterministic.
- **Supporting figures**:
  - timeline of sources by publication year and cluster (descending year,
    pre-2013 bucketed) — publication year is the reliable axis;
  - publication-type × cluster heatmap (with explicit axis labels and
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
- Clusters are computed from titles (plus the leading clause of a venue for
  untitled sources), not abstracts. The partition is strong at the cluster
  level (modularity ≈ 0.59), but a given short title can fall in an adjacent
  cluster; the cluster *labels* are the trustworthy unit, not any single
  node's colour. The clustering parameters (k, threshold, resolution) shape the
  granularity; they are documented constants in `build_figures.py` and their
  effect is reproducible.
- Text payloads are deliberately length-controlled. A short policy summary
  and a long technical paper are treated as comparable inputs; this avoids
  page-count dominance but flattens substantive depth.
- The figure should be read as evidence about *our reading list*, not as a
  ground-truth map of an evolving field. Patterns it surfaces (e.g.
  governance over-representation) are findings to comment on, not flaws.

## References

- Bertin, J. (1967) *Sémiologie graphique*. Paris: Mouton.
- Blei, D.M., Ng, A.Y. and Jordan, M.I. (2003) ‘Latent Dirichlet allocation’, *Journal of Machine Learning Research*, 3, pp. 993–1022.
- Blondel, V.D., Guillaume, J.-L., Lambiotte, R. and Lefebvre, E. (2008) ‘Fast unfolding of communities in large networks’, *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008.
- Brewer, C.A. (1994) ‘Color use guidelines for mapping and visualization’, in MacEachren, A.M. and Taylor, D.R.F. (eds.) *Visualization in Modern Cartography*. Oxford: Pergamon, pp. 123–147.
- Chen, C. (2006) ‘CiteSpace II: Detecting and visualizing emerging trends and transient patterns in scientific literature’, *Journal of the American Society for Information Science and Technology*, 57(3), pp. 359–377.
- Fruchterman, T.M.J. and Reingold, E.M. (1991) ‘Graph drawing by force-directed placement’, *Software: Practice and Experience*, 21(11), pp. 1129–1164.
- Grootendorst, M. (2022) ‘BERTopic: Neural topic modeling with a class-based TF-IDF procedure’, *arXiv preprint* arXiv:2203.05794.
- Kessler, M.M. (1963) ‘Bibliographic coupling between scientific papers’, *American Documentation*, 14(1), pp. 10–25.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S. and Dean, J. (2013) ‘Distributed representations of words and phrases and their compositionality’, in *Advances in Neural Information Processing Systems 26*.
- Newman, M.E.J. (2006) ‘Modularity and community structure in networks’, *Proceedings of the National Academy of Sciences*, 103(23), pp. 8577–8582.
- Page, M.J. et al. (2021) ‘The PRISMA 2020 statement: An updated guideline for reporting systematic reviews’, *BMJ*, 372, n71.
- Reimers, N. and Gurevych, I. (2019) ‘Sentence-BERT: Sentence embeddings using Siamese BERT-networks’, *EMNLP 2019*.
- Rethlefsen, M.L. et al. (2021) ‘PRISMA-S: An extension to the PRISMA statement for reporting literature searches in systematic reviews’, *Systematic Reviews*, 10(1), 39.
- Salton, G. and Buckley, C. (1988) ‘Term-weighting approaches in automatic text retrieval’, *Information Processing & Management*, 24(5), pp. 513–523.
- Salton, G. and McGill, M.J. (1983) *Introduction to Modern Information Retrieval*. New York: McGraw-Hill.
- Small, H. (1973) ‘Co-citation in the scientific literature: A new measure of the relationship between two documents’, *Journal of the American Society for Information Science*, 24(4), pp. 265–269.
- van Eck, N.J. and Waltman, L. (2010) ‘Software survey: VOSviewer, a computer program for bibliometric mapping’, *Scientometrics*, 84(2), pp. 523–538.
