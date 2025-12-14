
"""
Model loading and semantic classification utilities.

This module is responsible for:
- Loading spaCy and sentence-transformers models
- Defining prototype phrases for statement / relation types
- Providing helpers to classify sentences and relations via cosine similarity
"""

from typing import Dict, List, Tuple
import numpy as np

try:
    import spacy
except ImportError as e:  # pragma: no cover - import-time error path
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    SentenceTransformer = None


# ---- Prototype phrases ------------------------------------------------------

CATEGORY_PROTOTYPES: Dict[str, List[str]] = {
    "precept": [
        "from now on assume that",
        "in this conversation we will treat",
        "let us define that",
        "whenever I say, interpret it as",
        "I want you to always",
    ],
    "value": [
        "it is important that",
        "what matters most is",
        "I care deeply about",
        "this should never happen",
    ],
    "fact": [
        "in reality it is the case that",
        "the fact is that",
        "we observe that",
        "data shows that",
    ],
    "preference": [
        "I prefer it when",
        "I like it when",
        "I would rather",
        "my favorite way is",
    ],
    "self_state": [
        "I feel like",
        "I think that",
        "lately I have been feeling",
        "I have the impression that",
    ],
}

REL_TEMPLATES: Dict[str, List[str]] = {
    "causal": [
        "X causes Y",
        "X leads to Y",
        "X results in Y",
        "when X happens, Y happens",
    ],
    "preference": [
        "X is better than Y",
        "I prefer X over Y",
        "X is more important than Y",
    ],
    "definition": [
        "X is defined as Y",
        "X means the same as Y",
        "X is another word for Y",
    ],
    "negation": [
        "X cannot be Y",
        "X is not the same as Y",
        "X and Y cannot both be true",
    ],
}


PRECEPT_MARKERS = {
    "from now on",
    "in this conversation",
    "assume that",
    "let's assume",
    "i want you to",
}


# ---- Token normalization ----------------------------------------------------

def normalize_token_id(token) -> str:
    """
    Map a spaCy token to a canonical node ID.

    - Pronouns collapse to coarse roles to keep cross-sentence identity
      (self/you/we/they).
    - Everything else uses lemma().lower().
    """
    txt = token.text.lower()
    if token.pos_ == "PRON":
        if txt in {"i", "me", "my", "mine", "myself"}:
            return "self"
        if txt in {"you", "your", "yours", "yourself"}:
            return "you"
        if txt in {"we", "us", "our", "ours", "ourselves"}:
            return "we"
        if txt in {"they", "them", "their", "theirs", "themselves"}:
            return "they"
    return token.lemma_.lower()


def load_models(
    spacy_model: str = "en_core_web_sm",
    sbert_model: str = "all-MiniLM-L6-v2",
):
    """
    Load spaCy and sentence-transformers models.

    Returns
    -------
    nlp: spacy.language.Language
        The loaded spaCy pipeline.
    sbert: SentenceTransformer
        The loaded sentence-transformers model.
    """
    if spacy is None:
        raise ImportError("spaCy is not installed. Please `pip install spacy`.")
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. Please `pip install sentence-transformers`."
        )

    nlp = spacy.load(spacy_model)
    sbert = SentenceTransformer(sbert_model)
    return nlp, sbert


# ---- Embedding helpers ------------------------------------------------------

def build_category_vectors(model, proto_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Average prototype embeddings per category and normalize.

    Parameters
    ----------
    model : SentenceTransformer
    proto_dict : dict[str, list[str]]

    Returns
    -------
    dict[str, np.ndarray]
    """
    cat_vecs: Dict[str, np.ndarray] = {}
    for cat, phrases in proto_dict.items():
        emb = model.encode(phrases, normalize_embeddings=True)
        cat_vecs[cat] = emb.mean(axis=0)
    return cat_vecs


def classify_sentence_semantic(
    text: str,
    model,
    cat_vecs: Dict[str, np.ndarray],
    min_conf: float = 0.35,
) -> Tuple[str, float]:
    """
    Classify a sentence into a coarse category via cosine similarity to
    prototype phrase embeddings.

    Parameters
    ----------
    text : str
        The sentence to classify.
    model : SentenceTransformer
    cat_vecs : dict[str, np.ndarray]
    min_conf : float
        Minimum cosine similarity to accept a label.

    Returns
    -------
    (category, score)
    """
    v = model.encode(text, normalize_embeddings=True)
    best_cat, best_score = None, -1.0
    for cat, cvec in cat_vecs.items():
        score = float(np.dot(v, cvec))
        if score > best_score:
            best_cat, best_score = cat, score
    if best_score < min_conf:
        return "unknown", best_score
    return best_cat or "unknown", best_score


def lexical_precept_hint(text: str) -> bool:
    """
    Quick lexical hint that something might be a precept-like statement.
    """
    lower = text.lower()
    return any(m in lower for m in PRECEPT_MARKERS)


def classify_statement(
    text: str,
    model,
    cat_vecs: Dict[str, np.ndarray],
) -> Tuple[str, float]:
    """
    Blend semantic classification with lexical hints to decide what kind
    of statement this is (precept, value, fact, preference, ...).

    Returns
    -------
    (label, confidence)
    """
    cat, score = classify_sentence_semantic(text, model, cat_vecs)

    # Lexical override for strong signals
    if lexical_precept_hint(text):
        return "precept", max(score, 0.8)

    return cat, score


def classify_relation(
    src_label: str,
    verb: str,
    dst_label: str,
    model,
    rel_vecs: Dict[str, np.ndarray],
    min_conf: float = 0.3,
) -> Tuple[str, float]:
    """
    Classify a relation between two concepts (src, dst) mediated by a verb
    into coarse semantic types like causal, preference, definition, negation.

    Parameters
    ----------
    src_label : str
    verb : str
    dst_label : str
    model : SentenceTransformer
    rel_vecs : dict[str, np.ndarray]
    min_conf : float

    Returns
    -------
    (relation_type, score)
    """
    text = f"{src_label} {verb} {dst_label}"
    v = model.encode(text, normalize_embeddings=True)
    best_rel, best_score = None, -1.0
    for rel_type, rvec in rel_vecs.items():
        score = float(np.dot(v, rvec))
        if score > best_score:
            best_rel, best_score = rel_type, score
    if best_score < min_conf:
        return "unknown", best_score
    return best_rel or "unknown", best_score
