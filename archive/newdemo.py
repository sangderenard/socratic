"""
Incremental Socratic Precept Graph Demo

Legacy exploratory demo retained for reference. Preferred entrypoint is the
library API (`socratic_graph.graph_with_embeddings` and related helpers).

- Takes a multi-sentence input text.
- For each sentence, it:
  - Classifies the sentence type (precept/value/fact/preference/self_state/unknown)
    using a sentence-transformers embedding vs prototype phrases.
  - Breaks the sentence into a crude concept graph using spaCy POS/dep parsing.
  - Classifies subject-verb-object relations into coarse types (causal/definition/preference/negation/unknown).
  - Merges nodes into a global graph using a normalized "platonic" node ID so
    the same word (lemma) becomes the same node across sentences.
  - Prints the nodes/edges contributed by that sentence.
  - Draws the entire graph so far.

Node deduplication:
- We normalize tokens by lemma().lower()
- Pronouns like I/me/my/mine/myself → "self"
- You/your/yours/yourself → "you"
"""
import argparse
import os
import random

import nltk
from nltk.corpus import gutenberg

import argparse
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import spacy
from sentence_transformers import SentenceTransformer
# -------------------------------------------------------------------------
# Gutenberg loader
# -------------------------------------------------------------------------

# Friendly short names → NLTK Gutenberg file IDs
GUTENBERG_CHOICES = {
    "emma": "austen-emma.txt",
    "persuasion": "austen-persuasion.txt",
    "sense": "austen-sense.txt",
    "bible": "bible-kjv.txt",
    "moby": "melville-moby_dick.txt",
    "caesar": "shakespeare-caesar.txt",
    "hamlet": "shakespeare-hamlet.txt",
    "macbeth": "shakespeare-macbeth.txt",
}


def ensure_gutenberg_downloaded() -> None:
    """Make sure the Gutenberg corpus (and punkt) are available."""
    try:
        _ = gutenberg.fileids()
    except LookupError:
        nltk.download("gutenberg")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def load_gutenberg_text(choice: str | None = None) -> tuple[str, str]:
    """
    Load text from the NLTK Gutenberg corpus.

    Parameters
    ----------
    choice : str | None
        - None or 'RANDOM' → pick a random file from GUTENBERG_CHOICES.
        - If matches a key in GUTENBERG_CHOICES → use that file.
        - If matches an exact gutenberg file ID → use that directly.

    Returns
    -------
    (text, file_id)
    """
    ensure_gutenberg_downloaded()

    # Available IDs
    fileids = set(gutenberg.fileids())

    if choice is None or choice.upper() == "RANDOM":
        # restrict random to our curated set, but only those actually present
        available = [fid for fid in GUTENBERG_CHOICES.values() if fid in fileids]
        if not available:
            # fallback: any gutenberg file
            available = list(fileids)
        file_id = random.choice(available)
        text = gutenberg.raw(file_id)
        return text, file_id

    # Normalize user choice
    key = choice.lower()

    # If user passed a friendly short name
    if key in GUTENBERG_CHOICES:
        file_id = GUTENBERG_CHOICES[key]
        if file_id not in fileids:
            raise SystemExit(f"Gutenberg file not found in corpus: {file_id}")
        text = gutenberg.raw(file_id)
        return text, file_id

    # If user passed an exact fileid (e.g. 'shakespeare-macbeth.txt')
    if choice in fileids:
        text = gutenberg.raw(choice)
        return text, choice

    raise SystemExit(
        f"Unknown Gutenberg choice: {choice!r}\n"
        f"Known short names: {sorted(GUTENBERG_CHOICES.keys())}\n"
        f"Or pass an exact gutenberg fileid from: {sorted(fileids)}"
    )


# -----------------------------------------------------------------------------
# Prototype phrases for sentence-level classification
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Model loading and embedding helpers
# -----------------------------------------------------------------------------

def load_models(
    spacy_model: str = "en_core_web_sm",
    sbert_model: str = "all-MiniLM-L6-v2",
):
    """Load spaCy and sentence-transformers models."""
    nlp = spacy.load(spacy_model)
    sbert = SentenceTransformer(sbert_model)
    return nlp, sbert


def build_category_vectors(model, proto_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Average prototype embeddings per category and normalize."""
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
    """Classify a sentence into a coarse category by cosine similarity."""
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
    """Quick lexical hint that something might be a precept-like statement."""
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
    """
    cat, score = classify_sentence_semantic(text, model, cat_vecs)

    # Lexical override for strong precept phrases
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


# -----------------------------------------------------------------------------
# Graph construction helpers (per sentence)
# -----------------------------------------------------------------------------

def normalize_token_id(token) -> str:
    """
    Map a spaCy token to a canonical node ID.

    - Pronouns like I/me/my/mine/myself → "self"
    - You/your/yours/yourself → "you"
    - Otherwise lemma().lower()
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
    # default: lemma-based identity
    return token.lemma_.lower()


def sentence_to_graph_components(sent, sbert, rel_vecs) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, str, Dict]]]:
    """
    Given a spaCy Span (sentence), return:
    - nodes: list of (node_id, attrs_dict)
    - edges: list of (src_id, dst_id, attrs_dict)

    Uses very crude heuristics:
    - nouns/pronouns as concept nodes (dedup by normalized ID)
    - adjectives/adverbs as modifier nodes
    - subject-verb-object edges as relations
    """
    nodes: Dict[str, Dict] = {}
    edges: List[Tuple[str, str, Dict]] = []

    def add_node(key: str, **attrs):
        if key not in nodes:
            nodes[key] = {"label": key, **attrs}
        else:
            nodes[key].update(attrs)

    # Concept nodes
    for token in sent:
        if token.pos_ in ("NOUN", "PROPN", "PRON"):
            key = normalize_token_id(token)
            add_node(key, pos=token.pos_, kind="concept")

    # Modifier nodes (ADJ/ADV attached to concepts)
    for token in sent:
        if token.pos_ in ("ADJ", "ADV") and token.head.pos_ in ("NOUN", "PROPN", "PRON"):
            concept_key = normalize_token_id(token.head)
            # Modifier identity is "modifierlemma:conceptid" to keep it local
            mod_key = f"{token.lemma_.lower()}:{concept_key}"
            add_node(mod_key, pos=token.pos_, kind="modifier")
            edges.append((mod_key, concept_key, {"rel": "modifies"}))

    # Subject-verb-object relations
    for token in sent:
        if token.dep_ in ("ROOT", "conj") and token.pos_ == "VERB":
            verb = token
            verb_label = verb.lemma_.lower()

            subjects = [child for child in verb.children if child.dep_ in ("nsubj", "nsubjpass")]
            objects = [
                child for child in verb.children
                if child.dep_ in ("dobj", "pobj", "dative", "attr", "oprd")
            ]

            for subj in subjects:
                subj_key = normalize_token_id(subj)
                add_node(subj_key, pos=subj.pos_, kind="concept")

                if objects:
                    for obj in objects:
                        obj_key = normalize_token_id(obj)
                        add_node(obj_key, pos=obj.pos_, kind="concept")

                        rel_type, rel_conf = classify_relation(
                            subj.lemma_,
                            verb_label,
                            obj.lemma_,
                            sbert,
                            rel_vecs,
                        )
                        edges.append(
                            (
                                subj_key,
                                obj_key,
                                {
                                    "rel": verb_label,
                                    "rel_type": rel_type,
                                    "rel_conf": rel_conf,
                                },
                            )
                        )
                else:
                    # subject -> verb node if no explicit object
                    verb_key = f"verb:{verb_label}"
                    add_node(verb_key, pos=verb.pos_, kind="verb")
                    edges.append((subj_key, verb_key, {"rel": "acts"}))

    node_list = [(nid, attrs) for nid, attrs in nodes.items()]
    return node_list, edges


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def draw_graph(G: nx.DiGraph, title: str = "", figsize=(10, 8)) -> None:
    """
    Draw a networkx DiGraph using a spring layout.

    - Node color encodes kind (concept/modifier/sentence/verb)
    - Sentence nodes with stmt_type="precept" are red, other sentence nodes orange
    - Edge labels show raw relation and, if available, relation type
    """
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=0.7, iterations=100)

    node_colors = []
    node_labels = {}

    for n, data in G.nodes(data=True):
        kind = data.get("kind", "concept")
        stmt_type = data.get("stmt_type")

        if kind == "concept":
            node_colors.append("lightblue")
        elif kind == "modifier":
            node_colors.append("lightgreen")
        elif kind == "sentence":
            if stmt_type == "precept":
                node_colors.append("red")
            else:
                node_colors.append("orange")
        elif kind == "verb":
            node_colors.append("yellow")
        else:
            node_colors.append("gray")

        node_labels[n] = n

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")

    edge_labels = {}
    for u, v, data in G.edges(data=True):
        rel = data.get("rel", "")
        rel_type = data.get("rel_type", "")
        if isinstance(rel, list):
            rel = ",".join(sorted(set([r for r in rel if r])))
        if rel_type and rel_type != "unknown":
            label = f"{rel}/{rel_type}" if rel else rel_type
        else:
            label = rel
        edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

def build_full_graph_from_text(text: str) -> nx.DiGraph:
    """
    Build a single global graph from a raw text, treating periods as sentence
    delimiters.

    - Splits text on '.'
    - For each chunk, runs the same pipeline as in debug_incremental_demo
      (statement classification + POS/dep graph construction)
    - Deduplicates nodes via normalize_token_id (same as incremental mode)
    - Returns the final networkx.DiGraph
    """
    print("Loading models...")
    nlp, sbert = load_models()

    print("Building prototype vectors...")
    cat_vecs = build_category_vectors(sbert, CATEGORY_PROTOTYPES)
    rel_vecs = build_category_vectors(sbert, REL_TEMPLATES)

    G = nx.DiGraph()

    # Naive "period-to-period" segmentation
    raw_sents = [s.strip() for s in text.split(".") if s.strip()]

    for idx, sent_text in enumerate(raw_sents, start=1):
        # restore trailing period so it still looks like a sentence
        sent_text = sent_text + "."
        print(f"Processing sentence {idx}: {sent_text!r}")

        stmt_type, conf = classify_statement(sent_text, sbert, cat_vecs)

        # Parse this single sentence with spaCy
        sent_doc = nlp(sent_text)
        # There *should* be one sentence, but we iterate defensively
        for sent in sent_doc.sents:
            nodes, edges = sentence_to_graph_components(sent, sbert, rel_vecs)

            # Merge node attrs into global graph
            for nid, attrs in nodes:
                if nid in G.nodes:
                    G.nodes[nid].update(attrs)
                else:
                    G.add_node(nid, **attrs)

            # Merge edges; collapse multiple 'rel' values into a list
            for src, dst, attrs in edges:
                if G.has_edge(src, dst):
                    existing = G[src][dst].get("rel", [])
                    if not isinstance(existing, list):
                        existing = [existing]
                    if attrs.get("rel"):
                        existing.append(attrs["rel"])
                    G[src][dst]["rel"] = existing
                else:
                    G.add_edge(src, dst, **attrs)

            # Sentence node and "about" edges (same pattern as incremental)
            sent_id = f"sent:{idx}"
            G.add_node(
                sent_id,
                kind="sentence",
                stmt_type=stmt_type,
                stmt_conf=conf,
                text=sent_text,
            )

            main_concepts = [
                normalize_token_id(t) for t in sent if t.pos_ in ("NOUN", "PROPN")
            ]
            for c in main_concepts:
                if c in G.nodes:
                    G.add_edge(sent_id, c, rel="about")

    return G

# -----------------------------------------------------------------------------
# Incremental demo
# -----------------------------------------------------------------------------

def debug_incremental_demo(text: str) -> None:
    """
    For each sentence in the text:
    - classify and print its statement type
    - extract and print nodes/edges from that sentence
    - incrementally merge into a global graph
    - display the graph after each step
    """
    print("Loading models...")
    nlp, sbert = load_models()

    print("Building prototype vectors...")
    cat_vecs = build_category_vectors(sbert, CATEGORY_PROTOTYPES)
    rel_vecs = build_category_vectors(sbert, REL_TEMPLATES)

    doc = nlp(text)
    G = nx.DiGraph()

    for idx, sent in enumerate(doc.sents, start=1):
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        print("=" * 80)
        print(f"Sentence {idx}: {sent_text!r}")

        stmt_type, conf = classify_statement(sent_text, sbert, cat_vecs)
        print(f"  → stmt_type = {stmt_type!r}, confidence = {conf:.3f}")

        # Extract sentence-level graph components
        nodes, edges = sentence_to_graph_components(sent, sbert, rel_vecs)

        print("\n  Nodes from this sentence:")
        for nid, attrs in nodes:
            print(f"    {nid!r} -> {attrs}")

        print("\n  Edges from this sentence:")
        for src, dst, attrs in edges:
            print(f"    {src!r} -> {dst!r} : {attrs}")

        # Merge into global graph with deduplicated node IDs
        for nid, attrs in nodes:
            if nid in G.nodes:
                G.nodes[nid].update(attrs)
            else:
                G.add_node(nid, **attrs)

        for src, dst, attrs in edges:
            if G.has_edge(src, dst):
                existing = G[src][dst].get("rel", [])
                if not isinstance(existing, list):
                    existing = [existing]
                if attrs.get("rel"):
                    existing.append(attrs["rel"])
                G[src][dst]["rel"] = existing
            else:
                G.add_edge(src, dst, **attrs)

        # Add a sentence node referencing its concepts
        sent_id = f"sent:{idx}"
        G.add_node(
            sent_id,
            kind="sentence",
            stmt_type=stmt_type,
            stmt_conf=conf,
            text=sent_text,
        )

        # Connect sentence node to main concepts (lemma-based)
        main_concepts = [normalize_token_id(t) for t in sent if t.pos_ in ("NOUN", "PROPN")]
        for c in main_concepts:
            if c in G.nodes:
                G.add_edge(sent_id, c, rel="about")

        # Draw the global graph after this sentence
        title = f"Graph after sentence {idx}"
        print(f"\nDrawing graph: {title}")
        draw_graph(G, title=title)

    print("=" * 80)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incremental or full-graph Socratic precept visualization."
    )
    parser.add_argument(
        "-f",
        "--file",
        help=(
            "Path to a text file. If provided, build and display one full graph "
            "from the file (period-separated sentences)."
        ),
    )
    parser.add_argument(
        "-g",
        "--gutenberg",
        nargs="?",
        const="RANDOM",
        metavar="BOOK",
        help=(
            "Use an NLTK Gutenberg book instead of a local file. "
            "If BOOK is omitted, a random book is chosen from a small set "
            f"({', '.join(sorted(GUTENBERG_CHOICES.keys()))}). "
            "You can also pass a full gutenberg fileid like 'shakespeare-macbeth.txt'."
        ),
    )
    args = parser.parse_args()

    if args.gutenberg is not None:
        # --- Gutenberg mode: single graph from a Gutenberg text ---
        text, file_id = load_gutenberg_text(args.gutenberg)
        print(f"Loaded Gutenberg text: {file_id}")
        G = build_full_graph_from_text(text)
        title = f"Gutenberg: {file_id}"
        print("\nDrawing full graph...")
        draw_graph(G, title=title)

    elif args.file:
        # --- Full file mode: single graph ---
        if not os.path.exists(args.file):
            raise SystemExit(f"File not found: {args.file}")

        with open(args.file, "r", encoding="utf-8") as fh:
            file_text = fh.read()

        G = build_full_graph_from_text(file_text)
        title = f"Full graph from {os.path.basename(args.file)}"
        print("\nDrawing full graph...")
        draw_graph(G, title=title)

    else:
        # --- Default: incremental demo on SAMPLE_TEXT ---
        SAMPLE_TEXT = """
        In this conversation, truth means best concordance with observed data shape.
        I want you to prioritize internal consistency over comfort.
        Sometimes I want you to just make me feel better even if it breaks the rules.
        """
        debug_incremental_demo(SAMPLE_TEXT)
