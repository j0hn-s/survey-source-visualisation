"""Pipeline for the semantic source map.

Modules are deliberately small and ordered to mirror the five stages described
in docs/methodology.md:

    1. parse_references  - turn the bibliography into structured metadata
    2. acquire_text      - obtain the text payload appropriate to each source
    3. preprocess        - normalise text into a comparable form
    4. features          - TF-IDF and (optionally) sentence embeddings
    5. topic_assignment  - rule-based scoring against the controlled vocabulary
    6. graph             - construct the typed multi-edge graph
    7. visualise         - render the figure used in the paper
"""
