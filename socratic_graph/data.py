"""
Data utilities that keep visualization and physics concerns separate.

Provides helpers to build a socratic graph from text, attach per-node
SentenceTransformer embeddings as resting positions, and compute edge
length metadata before any physics-based layout runs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from .graph_builder import text_to_graph_with_types
from .models import normalize_token_id


def _embedding_text_for_node(node_id: str, attrs: Dict) -> str:
    """Choose the text to embed for a node (sentence -> full text, else label)."""
    if attrs.get("kind") == "sentence":
        return attrs.get("text", node_id)
    return attrs.get("label", node_id)


def compute_node_embeddings(G, sbert, normalize: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute a SentenceTransformer embedding for every node.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph produced by `text_to_graph_with_types`.
    sbert : SentenceTransformer
    normalize : bool
        Whether to L2-normalize embeddings.

    Returns
    -------
    dict[node_id, np.ndarray]
    """
    vectors: Dict[str, np.ndarray] = {}
    for nid, attrs in G.nodes(data=True):
        text = _embedding_text_for_node(nid, attrs)
        vec = sbert.encode(text, normalize_embeddings=normalize)
        vectors[nid] = np.asarray(vec, dtype=float)
    return vectors


def edge_distance_table(G, embeddings: Dict[str, np.ndarray]) -> List[Dict]:
    """
    Build an edge table that includes Euclidean distance between node embeddings.

    Returns list of dicts suitable for JSON export, preserving rel/rel_type.
    """
    rows: List[Dict] = []
    for u, v, data in G.edges(data=True):
        if u not in embeddings or v not in embeddings:
            continue
        src_vec = embeddings[u]
        dst_vec = embeddings[v]
        dist = float(np.linalg.norm(dst_vec - src_vec))
        rel_val = data.get("rel")
        rel_type = data.get("rel_type")
        k_attr = float(data.get("k", 1.0))
        # normalize rel to a printable string
        if isinstance(rel_val, list):
            rel_val = ",".join(sorted({r for r in rel_val if r}))
        rows.append(
            {
                "source": u,
                "target": v,
                "distance": dist,
                "rel": rel_val,
                "rel_type": rel_type,
                "k": k_attr,
            }
        )
    return rows


def graph_with_embeddings(text: str, nlp, sbert, normalize: bool = False):
    """
    Build the socratic graph plus embedding-based resting state and edge metadata.

    Returns
    -------
    (G, embeddings, edge_distances)
      G: networkx.DiGraph
      embeddings: dict[node_id -> np.ndarray]
      edge_distances: list of dicts with distance/rel/rel_type per edge
    """
    G = text_to_graph_with_types(text, nlp, sbert)
    node_vecs = compute_node_embeddings(G, sbert, normalize=normalize)
    edges = edge_distance_table(G, node_vecs)
    return G, node_vecs, edges
