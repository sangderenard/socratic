from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import nltk
from nltk.corpus import gutenberg

from socratic_graph import (
    load_models,
    text_to_graph_with_types,
    compute_node_embeddings,
    edge_distance_table,
)


# -------------------------------------------------------------------------
# Gutenberg helpers
# -------------------------------------------------------------------------

def ensure_gutenberg_downloaded() -> None:
    """Ensure Gutenberg corpus and punkt tokenizer are available."""
    try:
        _ = gutenberg.fileids()
    except LookupError:
        nltk.download("gutenberg")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def load_gutenberg_text(choice: Optional[str] = None) -> tuple[str, str]:
    """
    Load text from the NLTK Gutenberg corpus.

    choice:
      - None or "RANDOM" → pick a random fileid
      - exact Gutenberg fileid string
    """
    ensure_gutenberg_downloaded()
    fileids = set(gutenberg.fileids())

    if choice is None or choice.upper() == "RANDOM":
        file_id = random.choice(sorted(fileids))
        text = gutenberg.raw(file_id)
        return text, file_id

    if choice in fileids:
        text = gutenberg.raw(choice)
        return text, choice

    raise SystemExit(
        f"Unknown Gutenberg choice: {choice!r}\n"
        f"Available fileids include e.g.: {sorted(list(fileids))[:10]} ..."
    )


# -------------------------------------------------------------------------
# Graph → mass–spring system
# -------------------------------------------------------------------------

def sentence_order_key(nid: str) -> int:
    """Order sentence nodes by the integer suffix in 'sent:NNN'."""
    if nid.startswith("sent:"):
        try:
            return int(nid.split(":", 1)[1])
        except Exception:
            return 0
    return 0


def build_mass_spring_from_graph(
    G,
    sbert,
    k_seq: float = 1.0,
    k_edge: float = 0.5,
) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Dict], List[Tuple[int, int, float, float, str]]]:
    """
    Build a mass–spring system directly from the socratic_graph networkx graph G.

    - Sentence nodes (kind='sentence') get positions from SBERT sentence embeddings.
    - All other nodes get initial positions as the mean of connected sentence embeddings.
    - Springs:
       * 'seq' between consecutive sentence nodes:
           rest length = initial distance
           stiffness   = k_seq
       * 'assoc' between any edge touching a sentence node:
           rest length = 1.0
           stiffness   = k_edge

    Returns:
      X0          : (N, D) initial positions
      node_ids    : list of node ids (length N)
      node_kinds  : list of node kinds (parallel to node_ids)
      node_meta   : id → metadata (text, lemma, etc.)
      springs     : list of (i, j, k, rest_len, spring_type)
    """
    # Identify sentence nodes and others
    sentence_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "sentence"]
    sentence_nodes = sorted(sentence_nodes, key=sentence_order_key)
    other_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") != "sentence"]

    if not sentence_nodes:
        raise ValueError("Graph has no sentence nodes (kind='sentence').")

    # Build node ordering: sentences first, then others (stable, deterministic)
    node_ids = sentence_nodes + sorted(other_nodes)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    node_kinds = [G.nodes[nid].get("kind", "concept") for nid in node_ids]

    # Sentence embeddings
    sent_texts = [G.nodes[nid].get("text", "") for nid in sentence_nodes]
    sent_emb = sbert.encode(sent_texts, normalize_embeddings=False)
    sent_emb = np.asarray(sent_emb)
    num_sent, dim = sent_emb.shape

    # Initial positions
    X0 = np.zeros((len(node_ids), dim), dtype=np.float32)

    # Map sentence node → row index in sent_emb
    sent_idx_map = {nid: i for i, nid in enumerate(sentence_nodes)}

    # Sentence positions = embeddings
    for nid in sentence_nodes:
        i = node_index[nid]
        j = sent_idx_map[nid]
        X0[i] = sent_emb[j]

    # Global mean for fallback
    global_mean = sent_emb.mean(axis=0)

    # Other node positions = mean of connected sentence embeddings
    for nid in other_nodes:
        i = node_index[nid]
        sent_neighbors: List[str] = []

        # in-edges: u -> nid
        for u, _ in G.in_edges(nid):
            if G.nodes[u].get("kind") == "sentence":
                sent_neighbors.append(u)

        # out-edges: nid -> v
        for _, v in G.out_edges(nid):
            if G.nodes[v].get("kind") == "sentence":
                sent_neighbors.append(v)

        # Deduplicate neighbors
        sent_neighbors = list(dict.fromkeys(sent_neighbors))

        if sent_neighbors:
            idxs = [sent_idx_map[s] for s in sent_neighbors]
            X0[i] = sent_emb[idxs].mean(axis=0)
        else:
            # If no direct connection to any sentence, park near the global mean
            X0[i] = global_mean + 0.01 * np.random.randn(dim)

    # Springs
    springs: List[Tuple[int, int, float, float, str]] = []

    # Sequential springs (sentence chain)
    for idx in range(len(sentence_nodes) - 1):
        nid1 = sentence_nodes[idx]
        nid2 = sentence_nodes[idx + 1]
        i = node_index[nid1]
        j = node_index[nid2]
        diff = X0[j] - X0[i]
        dist = float(np.linalg.norm(diff))
        if dist < 1e-8:
            dist = 1e-3
        springs.append((i, j, k_seq, dist, "seq"))

    # Association springs: any edge touching a sentence node, unit rest length
    for u, v, data in G.edges(data=True):
        kind_u = G.nodes[u].get("kind")
        kind_v = G.nodes[v].get("kind")
        if kind_u == "sentence" or kind_v == "sentence":
            i = node_index[u]
            j = node_index[v]
            springs.append((i, j, k_edge, 1.0, "assoc"))

    # Node metadata for export
    node_meta: Dict[str, Dict] = {}
    for nid in node_ids:
        attrs = dict(G.nodes[nid])
        # normalize some keys for the export
        meta: Dict = {}
        if "text" in attrs:
            meta["text"] = attrs["text"]
        if "label" in attrs:
            meta["lemma"] = attrs["label"]
        if "stmt_type" in attrs:
            meta["stmt_type"] = attrs["stmt_type"]
        if "stmt_conf" in attrs:
            meta["stmt_conf"] = float(attrs["stmt_conf"])
        node_meta[nid] = meta

    return X0, node_ids, node_kinds, node_meta, springs


# -------------------------------------------------------------------------
# Mass–spring simulation and PCA projection
# -------------------------------------------------------------------------

def simulate_mass_spring(
    X0: np.ndarray,
    springs: List[Tuple[int, int, float, float, str]],
    steps: int = 200,
    dt: float = 0.01,
    damping: float = 0.05,
) -> np.ndarray:
    """
    Damped explicit mass–spring simulation.

    - X0: (N, D) initial positions
    - springs: (i, j, k, rest_len, spring_type)
    - masses = 1
    - velocities start at 0
    """
    X = X0.astype(np.float32).copy()
    V = np.zeros_like(X, dtype=np.float32)
    N, D = X.shape

    print(f"Simulating mass–spring system: N={N}, D={D}, steps={steps}, dt={dt}, damping={damping}")

    for step in range(steps):
        F = np.zeros_like(X, dtype=np.float32)

        for i, j, k, L0, stype in springs:
            diff = X[j] - X[i]
            dist = float(np.linalg.norm(diff))
            if dist < 1e-8:
                continue
            stretch = dist - L0
            force_vec = (k * stretch / dist) * diff
            F[i] += force_vec
            F[j] -= force_vec

        # Damping
        V *= (1.0 - damping)
        # Acceleration (mass = 1)
        V += dt * F
        # Position update
        X += dt * V

        if (step + 1) % max(1, steps // 10) == 0:
            print(f"  step {step+1}/{steps}")

    print("Simulation complete.")
    return X


def pca_to_3d(X: np.ndarray) -> np.ndarray:
    """
    Project X (N, D) to 3D via PCA.

    Returns coords (N, 3).
    """
    X = X.astype(np.float64)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:3].T  # (D, 3)
    coords3d = Xc @ P
    return coords3d.astype(np.float32)


def export_to_json(
    path: str,
    coords3d: np.ndarray,
    node_ids: List[str],
    node_kinds: List[str],
    node_meta: Dict[str, Dict],
    springs: List[Tuple[int, int, float, float, str]],
) -> None:
    """
    Export relaxed 3D coordinates + spring connectivity to a JSON file.
    """
    N, D = coords3d.shape
    assert D == 3

    nodes_out = []
    for idx, nid in enumerate(node_ids):
        kind = node_kinds[idx]
        meta = node_meta.get(nid, {})
        entry = {
            "id": nid,
            "kind": kind,
            "pos": coords3d[idx].tolist(),
        }
        entry.update(meta)
        nodes_out.append(entry)

    edges_out = []
    for i, j, k, L0, stype in springs:
        edges_out.append({
            "source": node_ids[i],
            "target": node_ids[j],
            "spring_type": stype,
            "k": k,
            "rest_len": L0,
        })

    data = {
        "nodes": nodes_out,
        "edges": edges_out,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(nodes_out)} nodes and {len(edges_out)} edges to {path}")


def export_prephysics_json(
    path: str,
    G,
    node_vecs: Dict[str, np.ndarray],
    edge_rows: List[Dict],
) -> None:
    """Export raw graph + embedding resting positions + edge distances."""
    nodes_out = []
    for nid, attrs in G.nodes(data=True):
        entry = {
            "id": nid,
            "kind": attrs.get("kind", "concept"),
            "embedding": node_vecs.get(nid, []).tolist(),
        }
        if "text" in attrs:
            entry["text"] = attrs["text"]
        if "label" in attrs:
            entry["lemma"] = attrs["label"]
        if "stmt_type" in attrs:
            entry["stmt_type"] = attrs["stmt_type"]
        if "stmt_conf" in attrs:
            entry["stmt_conf"] = float(attrs["stmt_conf"])
        nodes_out.append(entry)

    data = {
        "nodes": nodes_out,
        "edges": edge_rows,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Exported pre-physics graph to {path} ({len(nodes_out)} nodes, {len(edge_rows)} edges)")


# -------------------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build mass–spring layout from socratic_graph and export to JSON."
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to local text file."
    )
    parser.add_argument(
        "-g", "--gutenberg",
        nargs="?",
        const="RANDOM",
        metavar="FILEID",
        help=(
            "Use an NLTK Gutenberg text instead of a local file. "
            "If FILEID is omitted, choose a random one. "
            "Otherwise pass an exact gutenberg fileid, e.g. 'shakespeare-macbeth.txt'."
        ),
    )
    parser.add_argument(
        "-o", "--out",
        default="socratic_spring.json",
        help="Output JSON file (default: socratic_spring.json)."
    )
    parser.add_argument(
        "--pre-out",
        default="socratic_prephysics.json",
        help="Output JSON for pre-physics data (default: socratic_prephysics.json)."
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Limit to the first N sentences of the text (spaCy-based)."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of simulation steps (default: 200)."
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Simulation time step (default: 0.01)."
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.05,
        help="Velocity damping factor per step (0..1, default: 0.05)."
    )
    parser.add_argument(
        "--k-seq",
        type=float,
        default=1.0,
        help="Spring constant for sequential sentence springs (default: 1.0)."
    )
    parser.add_argument(
        "--k-edge",
        type=float,
        default=0.5,
        help="Spring constant for sentence-connected edge springs (default: 0.5)."
    )
    parser.add_argument(
        "--skip-physics",
        action="store_true",
        help="Only export pre-physics embeddings + edge distances and exit."
    )

    args = parser.parse_args()

    # 1) Load raw text
    if args.gutenberg is not None:
        text, fid = load_gutenberg_text(args.gutenberg)
        print(f"Loaded Gutenberg text: {fid}")
    elif args.file:
        if not os.path.exists(args.file):
            raise SystemExit(f"File not found: {args.file}")
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
        print(f"Loaded local file: {args.file}")
    else:
        text = """
        In this conversation, truth means best concordance with observed data shape.
        I want you to prioritize internal consistency over comfort.
        Sometimes I want you to just make me feel better even if it breaks the rules.
        """
        print("Using built-in SAMPLE_TEXT (pass --file or --gutenberg for real material).")

    # 2) Load models once
    print("Loading models (spaCy + SBERT)...")
    nlp, sbert = load_models()

    # 3) Optionally truncate to first N sentences using the same nlp
    if args.max_sentences is not None:
        doc = nlp(text)
        sents = []
        for i, sent in enumerate(doc.sents):
            if i >= args.max_sentences:
                break
            sents.append(sent.text)
        text = " ".join(sents)
        print(f"Truncated to first {len(sents)} sentences for graph building.")

    # 4) Build socratic graph
    print("Building socratic_graph network...")
    G = text_to_graph_with_types(text, nlp, sbert)

    # 5) Pre-physics export: embeddings + edge distances
    print("Computing pre-physics embeddings and edge distances...")
    node_vecs = compute_node_embeddings(G, sbert, normalize=False)
    edge_rows = edge_distance_table(G, node_vecs)
    if args.pre_out:
        export_prephysics_json(args.pre_out, G, node_vecs, edge_rows)

    if args.skip_physics:
        return

    # 6) Build mass–spring system from graph
    print("Constructing mass–spring system from graph...")
    X0, node_ids, node_kinds, node_meta, springs = build_mass_spring_from_graph(
        G,
        sbert,
        k_seq=args.k_seq,
        k_edge=args.k_edge,
    )

    # 7) Simulate
    X_final = simulate_mass_spring(
        X0,
        springs,
        steps=args.steps,
        dt=args.dt,
        damping=args.damping,
    )

    # 8) PCA to 3D
    coords3d = pca_to_3d(X_final)

    # 9) Export to JSON
    export_to_json(
        args.out,
        coords3d,
        node_ids,
        node_kinds,
        node_meta,
        springs,
    )


if __name__ == "__main__":
    main()
