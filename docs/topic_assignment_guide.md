# Topic assignment guide

This is the working procedure for turning machine-suggested topics into the
final values in `sources.csv`. It is the practical companion to the
methodological description in `methodology.md` §Stage 5.

## What you have in front of you

- `data/sources.csv` — one row per source, with `primary_topic` blank where a
  reviewer has not yet made a decision.
- `data/topic_suggestions.csv` — for every source, the top-scoring topic, the
  runner-up, both scores, the margin, and a `confident` flag.
- `data/abstracts_cache/{id}.txt` or `.pdf` — the text payload the suggestion
  was based on. **Always read it before accepting a suggestion.**

## The rule that decides the primary topic

The primary topic reflects the **central contribution** of the source, not
every theme it mentions. A paper that introduces a new FHE bootstrapping
trick and incidentally mentions a deployment is `mechanism`. A regulator's
guidance that recommends specific FHE deployments is `governance`.

When a source plausibly fits two topics, ask: which section of the survey
would substantively discuss it? Assign that topic. If you genuinely cannot
choose, use `secondary_topics` to capture the second one, and pick the
primary topic that aligns with the survey section where the source is
*introduced*.

## Workflow per source

1. **Read the entry in `raw`** (no inference from the title alone).
2. **Read the text payload** in `abstracts_cache/{id}.txt` if it exists.
3. **Open `topic_suggestions.csv`** and read the suggested topic, runner-up,
   and margin.
4. **Decide**:
   - If the suggestion matches your reading and `confident` is true,
     accept: copy the suggested topic into `primary_topic`. Leave
     `review_note` blank.
   - If the suggestion matches but `confident` is false (margin < 0.05),
     accept *and* record in `review_note`: e.g.
     `low-margin-accepted-suggestion: confirmed mechanism over systems`.
   - If you disagree with the suggestion, override `primary_topic` to your
     value *and* record the reason in `review_note`. Required form:
     `overridden: {suggested} → {chosen} ({short reason})`.
5. **PET families**: edit `pet_family` only when the seed value is wrong.
   Multi-label, `;`-separated. Add families the parser missed; remove
   families that only appear incidentally.

## How many sources need full review?

- Every source with `confident == False` in `topic_suggestions.csv`.
  Empirically this is roughly 10–20% of rows after abstracts have been
  acquired, and can be much higher when running on the title+venue fallback.
- A spot-check of at least 10% of `confident == True` sources, drawn at
  random and stratified by suggested topic.

## Auditability

The combination of `raw`, the cached payload, `topic_suggestions.csv`, and
the `review_note` column is intended to be enough that a reader who disputes
the figure can replay any individual decision. Do not edit suggestions
in place; always edit the corresponding `primary_topic` and document the
reason.
