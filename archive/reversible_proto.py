"""
Prototype: build a deterministic, reversible token graph from a hard-coded sentence
and feed it directly into the OpenGL/pygame physics loop.

- Uses SentenceTransformer embeddings so spring rest lengths come from the same n-D
    space the rest of the system uses.
- Nodes: one sentence anchor + one token node per surface token.
- Edges: sentence->token membership and token->token order chain.
- Deterministic linearization for round-trip text without any LM.
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple

import numpy as np

from gl_animator import run as run_animator
from socratic_graph import load_models


EXAMPLE_SENTENCE = "The quick brown fox jumps over the lazy dog."


def tokenize_with_whitespace(text: str) -> List[Tuple[str, str]]:
    """Return list of (token, trailing_ws) preserving punctuation and spacing."""
    tokens: List[Tuple[str, str]] = []
    for match in re.finditer(r"\S+|\s+", text):
        chunk = match.group(0)
        if chunk.isspace():
            if tokens:
                # append whitespace to previous token's trailing_ws
                prev_tok, prev_ws = tokens[-1]
                tokens[-1] = (prev_tok, prev_ws + chunk)
            continue
        # look ahead to capture following whitespace
        end = match.end()
        ws_match = re.match(r"\s+", text[end:])
        ws = ws_match.group(0) if ws_match else ""
        tokens.append((chunk, ws))
    return tokens


def build_reversible_graph(sentence: str, sbert) -> Tuple[List[Dict], List[Dict]]:
    toks = tokenize_with_whitespace(sentence)

    nodes: List[Dict] = []
    edges: List[Dict] = []

    # Embed sentence + tokens in one batch for consistent dimensionality
    batch_texts = [sentence] + [t for t, _ in toks]
    embeddings = sbert.encode(batch_texts, normalize_embeddings=False)
    sent_vec = embeddings[0]
    token_vecs = embeddings[1:]

    sent_id = "sent:0"
    nodes.append({"id": sent_id, "kind": "sentence", "text": sentence, "embedding": sent_vec.tolist()})

    for idx, ((tok, ws), vec) in enumerate(zip(toks, token_vecs)):
        nid = f"tok:0:{idx}"
        nodes.append({
            "id": nid,
            "kind": "token",
            "text": tok,
            "ws": ws,
            "embedding": vec.tolist(),
        })
        edges.append({"source": sent_id, "target": nid, "rel": "has_token", "k": 0.5})
        if idx > 0:
            prev_id = f"tok:0:{idx - 1}"
            edges.append({"source": prev_id, "target": nid, "rel": "next", "k": 1.0})

    return nodes, edges


def linearize(nodes: List[Dict]) -> str:
    """Deterministically rebuild text from token nodes using idx order."""
    toks = [n for n in nodes if n.get("kind") == "token"]
    # idx is encoded in id suffix
    toks = sorted(toks, key=lambda n: int(n["id"].split(":")[-1]))
    parts: List[str] = []
    for n in toks:
        parts.append(n.get("text", ""))
        parts.append(n.get("ws", ""))
    return "".join(parts)


def main():
    # Load models to obtain SentenceTransformer; spaCy is unused but load_models returns both
    _nlp, sbert = load_models()

    nodes, edges = build_reversible_graph(EXAMPLE_SENTENCE, sbert)
    regen = linearize(nodes)
    print("Original:", EXAMPLE_SENTENCE)
    print("Linearized:", regen)
    print("Launching animator… Close the window to exit.")

    run_animator(
        path=None,
        dt=0.01,
        k_spring=1.0,
        k_rep=0.0,
        k_angle=0.0,
        temp=0.0,
        damping=0.02,
        softening=1e-2,
        steps_per_frame=2,
        nodes_override=nodes,
        edges_override=edges,
    )


if __name__ == "__main__":
    main()
