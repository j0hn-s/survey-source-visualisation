"""Stage 4 - feature extraction.

Two complementary representations are produced for each source:

    1. TF-IDF (sparse, interpretable). We follow the formulation of
       Salton & Buckley (1988) - term frequency tempered by inverse
       document frequency. The choice is deliberate: TF-IDF lets us
       inspect which terms drive any pair of sources together, which is
       important for the auditability claim in the methodology.

    2. Sentence-Transformer embeddings (dense, semantic). Computed with
       sentence-transformers/all-MiniLM-L6-v2 (Reimers & Gurevych, 2019,
       Sentence-BERT). Used only for the optional semantic-similarity
       edge type. Disabling this representation is supported and
       leaves the rest of the pipeline working.

Why two? Bibliometric tooling (e.g. VOSviewer, van Eck & Waltman, 2010)
historically relied on bag-of-words measures alone; sentence embeddings
catch paraphrases (e.g. "executable enclave" / "trusted enclave") that
TF-IDF treats as unrelated. Reporting both keeps the figure honest:
if one edge set looks very different from the other, that is itself
informative about the evidence base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Features:
    ids: list[str]
    tfidf_matrix: object  # scipy.sparse.csr_matrix
    tfidf_vocab: list[str]
    embedding_matrix: Optional[np.ndarray] = None


def compute_tfidf(
    ids: list[str],
    texts: list[str],
    min_df: int = 2,
    max_df: float = 0.8,
    ngram_range: tuple[int, int] = (1, 2),
) -> Features:
    """Compute TF-IDF over the (already normalised) source texts.

    The defaults are chosen to:
      * drop hapax legomena (min_df=2) which add noise without informing
        similarity, following standard practice in bibliometric text mining;
      * drop terms appearing in >80% of documents (max_df=0.8) which are
        not discriminative;
      * include bigrams so that joined phrases like differential_privacy
        do not need to do all the work alone.
    """
    vec = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    mat = vec.fit_transform(texts)
    return Features(ids=ids, tfidf_matrix=mat, tfidf_vocab=vec.get_feature_names_out().tolist())


def compute_embeddings(
    ids: list[str],
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Compute sentence embeddings. Uses the raw (pre-normalisation) text.

    sentence-BERT was trained on natural-language sentences, so we feed the
    unnormalised acquired text rather than the bag-of-words form. Sources with
    empty payload get a zero vector and will therefore have zero cosine
    similarity to everything, which is the correct behaviour.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    safe_texts = [t if t.strip() else "" for t in texts]
    return model.encode(safe_texts, normalize_embeddings=True, show_progress_bar=False)


def pairwise_cosine(matrix) -> np.ndarray:
    """Cosine similarity matrix; accepts dense ndarray or scipy sparse."""
    return cosine_similarity(matrix)
