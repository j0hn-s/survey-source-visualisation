# Survey source visualisation

A reproducible pipeline that produces the semantic map of sources placed
between the Introduction and Section 2 of the PETs survey. The figure depicts
**what kinds of sources were considered, how they relate, and how they cluster
around the themes of the paper**. It is an analytical artefact, not decoration:
any visible imbalance is reported as a finding about the evidence base rather
than corrected away.

## What the figures show

The pipeline produces one main figure and three supporting figures.

**Main figure — semantic map** (`figures/semantic_map_main.png`)

- **Nodes**  — each cited source.
- **Node colour** — primary topic (controlled vocabulary, see [data/topic_vocabulary.yaml](data/topic_vocabulary.yaml)).
- **Node shape** — publication type (academic paper, preprint, survey, regulatory guidance, technical white paper, standards, industry blog, software repository).
- **Node size** — degree in the graph (a proxy for how many other sources it is related to).
- **Edges** — three independent rules, toggleable in the renderer:
  - shared primary topic;
  - shared PET family (homomorphic encryption, MPC, DP, FL, TEE, ZKP, synthetic data, syntactic anonymisation);
  - high TF-IDF cosine similarity, k-nearest-neighbour capped.

**Supporting figures**

- `figures/timeline.png` — sources by year and primary topic.
- `figures/type_topic_heatmap.png` — source type × primary topic crosstab with explicit axis labels.
- `figures/pet_family_breakdown.png` — counts across the seven PET families the
  survey explicitly considers (secure MPC, homomorphic encryption, differential
  privacy, synthetic data, zero-knowledge, federated learning and distributed
  analytics, trusted execution environments), stacked by primary topic. Sources
  that carry no tag from the seven are reported as a separate "Other / none"
  bar so they are visible rather than hidden.

## Repository layout

```
survey-source-visualisation/
├── README.md
├── reference_list.txt              # the bibliography (Cite Them Right Harvard)
├── requirements.txt
├── data/
│   ├── topic_vocabulary.yaml       # controlled vocabulary + heuristic patterns
│   ├── sources.csv                 # parser output, the audited source of truth
│   ├── source_text.csv             # text payloads (produced by acquire_text.py)
│   ├── topic_suggestions.csv       # ranked topic suggestions for human review
│   └── abstracts_cache/            # cached abstracts and human overrides
├── notebooks/
│   └── semantic_map.ipynb          # main visualisation notebook
├── src/
│   ├── parse_references.py         # Stage 1 - bibliography → metadata
│   ├── acquire_text.py             # Stage 2 - acquire text by source type
│   ├── preprocess.py               # Stage 3 - normalise text
│   ├── features.py                 # Stage 4 - TF-IDF and embeddings
│   ├── topic_assignment.py         # Stage 5 - rule-based topic scoring
│   ├── graph.py                    # Stage 6 - typed multi-edge graph
│   └── visualise.py                # Stage 7 - matplotlib + pyvis rendering
├── docs/
│   ├── methodology.md              # the long-form methodology
│   ├── data_dictionary.md          # column-by-column reference for sources.csv
│   └── topic_assignment_guide.md   # how a reviewer confirms or overrides topics
└── figures/                        # rendered output (gitignored)
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Stage 1 - parse the bibliography into structured metadata
python -m src.parse_references

# Stage 2 - acquire text (CrossRef + arXiv for abstracts; place
#           human-curated payloads in data/abstracts_cache/{id}.txt
#           for sources where automated retrieval is unavailable)
python -m src.acquire_text

# Stages 3-7 - run the notebook end-to-end
jupyter lab notebooks/semantic_map.ipynb
```

The notebook **runs even before Stage 2 has completed** and falls back to a
title+venue payload; it announces the fallback explicitly so a draft figure is
never mistaken for the published one.

## Methodology in brief

The full write-up lives in [docs/methodology.md](docs/methodology.md). In short, the pipeline is a
five-decision sequence — each decision is documented so a reader can reproduce
or contest it.

1. **Source inventory and classification** — every source gets one publication
   type, which determines the text-acquisition strategy.
2. **Text acquisition** — abstracts for academic and preprint sources;
   executive summaries (or first ~1,500 words) for long regulatory and white-
   paper documents; full text for short industry blogs. This deliberately
   normalises payload length across source types so that a 200-page technical
   guideline does not dominate a 4-page conference paper.
3. **Normalisation** — lower-case, lemmatise, stop-word removal, multi-word
   PET phrase atomisation (`differential privacy` → `differential_privacy`).
4. **Topic assignment** — hybrid. Mechanical TF-IDF scoring against a closed
   controlled vocabulary surfaces candidate topics; a human reviewer confirms
   or overrides them. Every override is recorded in the `review_note` column.
5. **Graph and visualisation** — three edge rules, Fruchterman-Reingold layout,
   colour-blind-safe palette.

## Where the logic is borrowed from

The pipeline draws on a small number of well-established methodological
precedents. They are cited inline in the source modules; consolidated here for
the reader.

| Step | Borrowed from |
| --- | --- |
| Bag-of-words / TF-IDF representation | Salton & Buckley (1988), *Term-weighting approaches in automatic text retrieval* |
| Bibliographic-coupling-style edges | Kessler (1963), *Bibliographic coupling between scientific papers* |
| Co-occurrence cluster edges | Small (1973), *Co-citation in the scientific literature* |
| Atomic multi-word phrases | Mikolov et al. (2013), *Distributed representations of words and phrases* (used here over a curated phrase list rather than a frequency threshold) |
| Sentence-Transformer embeddings | Reimers & Gurevych (2019), *Sentence-BERT* |
| k-NN edge cap on association strength | van Eck & Waltman (2010), *Software survey: VOSviewer* |
| Force-directed network layout | Fruchterman & Reingold (1991), *Graph drawing by force-directed placement* |
| Bibliometric network framing | Chen (2006), *CiteSpace II* |
| Human-in-the-loop transparency | Page et al. (2021), *PRISMA 2020*; Rethlefsen et al. (2021), *PRISMA-S* |
| Colour-blind-safe categorical palette | Brewer (1994), ColorBrewer |

The closed topic vocabulary (mechanism / systems / evaluation / assurance /
governance / deployment / survey) is **author-defined** — it mirrors the
structure of the survey paper rather than borrowing an existing taxonomy. This
choice, and its consequences, are discussed in [docs/methodology.md](docs/methodology.md).

## Limitations

- The figure reflects **the sources considered in this survey**, not the
  entirety of the PETs literature.
- Topic assignment is **guided, not unsupervised**: a reader of the figure is
  reading our taxonomy.
- Text payloads are deliberately length-controlled, so a one-line policy
  statement and a deeply technical paper are treated as comparable inputs.

If the figure surfaces a bias (e.g. an over-representation of governance and
DP-focused sources), this is treated as a finding to be commented on later in
the paper, not a flaw to be hidden.

## Tests

A pytest suite under [tests/](tests/) covers the deterministic parts of the
pipeline:

```bash
.venv/bin/python -m pytest tests/
```

The suite exercises the reference parser against representative entries from
the live bibliography (typographic quotes, arXiv preprints, year-with-suffix
patterns, ISO standards, software repositories), the topic-assignment scoring
on hand-crafted texts whose topical home is unambiguous, all three graph edge
rules in isolation, and the four rendering primitives as smoke tests. Network-
dependent stages (CrossRef / arXiv) and sentence-transformer embeddings are
integration concerns and are not covered here.
