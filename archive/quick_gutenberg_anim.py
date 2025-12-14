"""
Convenience runner: grab 64 sentences from a random NLTK Gutenberg text,
build pre-physics graph data, and launch the OpenGL/pygame animator.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import tempfile
import threading
import time
import math
from collections import Counter

import numpy as np
try:
    import torch
except Exception:  # torch optional
    torch = None

import nltk
from nltk.corpus import gutenberg

from socratic_graph import (
    load_models,
    graph_with_embeddings,
    suggest_simple_words_for_nodes,
    load_gpt2_small,
    generate_sentences_from_graph,
    deterministic_sentences_from_graph,
    load_wordfreq_lexicon,
    load_bert_fill_mask,
    fill_mask_sentences_from_graph,
)
from gl_animator import run as run_animator


def build_token_graph(text: str, nlp, sbert, max_sentences: int | None = None):
    """Build a reversible token graph with sentence and token nodes using SBERT embeddings."""
    doc = nlp(text)
    sents = list(doc.sents)
    if max_sentences is not None:
        sents = sents[:max_sentences]
    nodes: list[dict] = []
    edges: list[dict] = []

    sentence_texts = [s.text for s in sents]
    if not sentence_texts:
        return nodes, edges

    sent_emb = sbert.encode(sentence_texts, normalize_embeddings=False)

    sent_tokens: list[list] = [list(s) for s in sents]
    token_embeds: list[list[np.ndarray]] = []
    for toks in sent_tokens:
        if toks:
            vecs = sbert.encode([t.text for t in toks], normalize_embeddings=False)
            token_embeds.append([vec for vec in vecs])
        else:
            token_embeds.append([])

    sent_ids: list[str] = []
    sent_proj_ids: list[str] = []
    for si, (sent_text, vec) in enumerate(zip(sentence_texts, sent_emb)):
        sid = f"sent:{si}"
        spid = f"sentp:{si}"
        sent_ids.append(sid)
        sent_proj_ids.append(spid)
        base = {"text": sent_text, "embedding": vec.tolist()}
        nodes.append({"id": sid, "kind": "sentence", **base})
        nodes.append({"id": spid, "kind": "sentence_proj", **base, "fixed": True})

    ribbon_pairs: list[tuple[str, str]] = []

    for si, toks in enumerate(sent_tokens):
        sent_id = sent_ids[si]
        sent_proj_id = sent_proj_ids[si]
        vecs = token_embeds[si]
        first_token_id = None
        prev_tok_id = None
        for ti, tok in enumerate(toks):
            nid = f"tok:{si}:{ti}"
            if first_token_id is None:
                first_token_id = nid
            vec = vecs[ti] if ti < len(vecs) else sbert.encode(tok.text, normalize_embeddings=False)
            nodes.append({
                "id": nid,
                "kind": "token",
                "text": tok.text,
                "ws": tok.whitespace_,
                "embedding": vec.tolist(),
            })
            edges.append({"source": sent_id, "target": nid, "rel": "has_token", "k": 0.5})
            edges.append({"source": sent_proj_id, "target": nid, "rel": "has_token", "k": 0.5, "physics": False})
            if prev_tok_id is not None:
                edges.append({"source": prev_tok_id, "target": nid, "rel": "next", "k": 1.0})
            prev_tok_id = nid
        if si > 0 and first_token_id:
            prev_sent_tokens = sent_tokens[si - 1]
            if prev_sent_tokens:
                prev_last = f"tok:{si-1}:{len(prev_sent_tokens) - 1}"
                ribbon_pairs.append((prev_last, first_token_id))

    for i in range(len(sent_ids) - 1):
        edges.append({"source": sent_ids[i], "target": sent_ids[i + 1], "rel": "next_sentence", "k": 2.0})
        edges.append({"source": sent_proj_ids[i], "target": sent_proj_ids[i + 1], "rel": "next_sentence_proj", "k": 2.0, "physics": False})

    for src, dst in ribbon_pairs:
        edges.append({"source": src, "target": dst, "rel": "ribbon", "k": 1.0})

    return nodes, edges


def build_skipgram_pairs(token_order: list[str], id_to_idx: dict[str, int], window: int = 2) -> tuple[list[int], list[int], list[float]]:
    """Return (src_idx, dst_idx, weight) lists for skip-gram within window."""
    srcs: list[int] = []
    dsts: list[int] = []
    ws: list[float] = []
    n = len(token_order)
    for i, tid in enumerate(token_order):
        src_idx = id_to_idx.get(tid)
        if src_idx is None:
            continue
        for off in range(1, window + 1):
            j = i + off
            if j >= n:
                break
            dst_tid = token_order[j]
            dst_idx = id_to_idx.get(dst_tid)
            if dst_idx is None:
                continue
            srcs.append(src_idx)
            dsts.append(dst_idx)
            ws.append(1.0)
    return srcs, dsts, ws


def build_global_ngram_pairs(
    token_texts: list[str],
    node_text_to_idx: dict[str, list[int]],
    window: int = 2,
    max_pairs: int | None = 50000,
    exclude_pairs: set[tuple[str, str]] | None = None,
):
    """Build positive pairs from full-corpus token adjacency by surface form.

    token_texts: full-corpus token strings
    node_text_to_idx: mapping from token text -> list of node indices in current graph
    window: inclusive lookahead distance considered positive
    max_pairs: cap to avoid blow-up
    """
    srcs: list[int] = []
    dsts: list[int] = []
    ws: list[float] = []
    if not token_texts or not node_text_to_idx:
        return srcs, dsts, ws

    # Build adjacency from corpus surface forms
    adj: dict[str, set[str]] = {}
    n = len(token_texts)
    for i, tok in enumerate(token_texts):
        for off in range(1, window + 1):
            j = i + off
            if j >= n:
                break
            nxt = token_texts[j]
            if exclude_pairs and (tok, nxt) in exclude_pairs:
                continue
            if tok not in adj:
                adj[tok] = set()
            adj[tok].add(nxt)

    # Map to node indices
    total = 0
    for t, nbrs in adj.items():
        src_nodes = node_text_to_idx.get(t)
        if not src_nodes:
            continue
        for nbr in nbrs:
            dst_nodes = node_text_to_idx.get(nbr)
            if not dst_nodes:
                continue
            for si in src_nodes:
                for dj in dst_nodes:
                    srcs.append(si)
                    dsts.append(dj)
                    ws.append(1.0)
                    total += 1
                    if max_pairs is not None and total >= max_pairs:
                        return srcs, dsts, ws
    return srcs, dsts, ws


def _build_sim_tensors(nodes: list[dict], edges: list[dict]):
    node_ids = [n["id"] for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    embeds = []
    for n in nodes:
        emb = np.asarray(n.get("embedding", []), dtype=np.float32)
        embeds.append(emb)
    pos = np.vstack(embeds).astype(np.float32)
    springs = []
    for e in edges:
        if e.get("physics") is False or e.get("force") is False:
            continue
        s = e.get("source")
        t = e.get("target")
        if s not in id_to_idx or t not in id_to_idx:
            continue
        i = id_to_idx[s]
        j = id_to_idx[t]
        rest = float(np.linalg.norm(pos[i] - pos[j]))
        k_edge = float(e.get("k", 1.0))
        springs.append((i, j, rest, k_edge))
    springs = np.asarray(springs, dtype=np.float32) if springs else np.zeros((0, 4), dtype=np.float32)

    # angle neighbor triples (hub, n1, n2)
    neighbors = [[] for _ in range(len(node_ids))]
    for i, j, _, _ in springs:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))
    pair_n1 = []
    pair_n2 = []
    pair_hub = []
    for hub, neigh in enumerate(neighbors):
        deg = len(neigh)
        if deg < 2:
            continue
        for a in range(deg):
            for b in range(a + 1, deg):
                pair_n1.append(neigh[a])
                pair_n2.append(neigh[b])
                pair_hub.append(hub)

    return pos, springs, np.asarray(pair_n1, dtype=np.int64), np.asarray(pair_n2, dtype=np.int64), np.asarray(pair_hub, dtype=np.int64)


def train_k_params(nodes, edges, skip_src, skip_dst, skip_w, device: str, iters: int, lr: float, dt: float, k_spring_init: float, k_rep_init: float, k_angle_init: float):
    if torch is None:
        raise SystemExit("torch required for train_k_params")
    pos_np, springs_np, pair_n1_np, pair_n2_np, pair_hub_np = _build_sim_tensors(nodes, edges)
    if pos_np.size == 0 or springs_np.size == 0:
        print("[train-k] nothing to train (no positions or springs)")
        return None
    pos0 = torch.tensor(pos_np, dtype=torch.float32, device=device)
    springs = torch.tensor(springs_np, dtype=torch.float32, device=device)
    i_idx = springs[:, 0].long()
    j_idx = springs[:, 1].long()
    rest = springs[:, 2]
    edge_k = springs[:, 3]

    pair_n1 = torch.tensor(pair_n1_np, dtype=torch.long, device=device) if pair_n1_np.size else None
    pair_n2 = torch.tensor(pair_n2_np, dtype=torch.long, device=device) if pair_n2_np.size else None
    pair_hub = torch.tensor(pair_hub_np, dtype=torch.long, device=device) if pair_hub_np.size else None

    skip_src_t = torch.tensor(skip_src, dtype=torch.long, device=device) if skip_src else None
    skip_dst_t = torch.tensor(skip_dst, dtype=torch.long, device=device) if skip_dst else None
    skip_w_t = torch.tensor(skip_w, dtype=torch.float32, device=device) if skip_w else None
    if skip_src_t is None or skip_src_t.numel() == 0:
        print("[train-k] no skip-gram pairs; skipping training")
        return None

    k_spring_p = torch.nn.Parameter(torch.tensor(k_spring_init, dtype=torch.float32, device=device))
    k_rep_p = torch.nn.Parameter(torch.tensor(k_rep_init, dtype=torch.float32, device=device))
    k_angle_p = torch.nn.Parameter(torch.tensor(k_angle_init, dtype=torch.float32, device=device))

    opt = torch.optim.Adam([k_spring_p, k_rep_p, k_angle_p], lr=lr)

    for step in range(iters):
        pos = pos0.clone()
        vel = torch.zeros_like(pos)
        k_spring = torch.nn.functional.softplus(k_spring_p) + 1e-4
        k_rep = torch.nn.functional.softplus(k_rep_p)
        k_angle = torch.nn.functional.softplus(k_angle_p)

        # spring forces
        diff = pos[j_idx] - pos[i_idx]
        dist = torch.linalg.norm(diff, dim=1) + 1e-9
        stretch = rest - dist
        force_mag = (k_spring * edge_k) * stretch / dist
        F_edge = diff * (force_mag / dist)[:, None]
        F = torch.zeros_like(pos)
        F.index_add_(0, i_idx, -F_edge)
        F.index_add_(0, j_idx, F_edge)

        # angle spreading (optional)
        if k_angle.item() > 0 and pair_n1 is not None and pair_n1.numel():
            v1 = pos[pair_n1] - pos[pair_hub]
            v2 = pos[pair_n2] - pos[pair_hub]
            norm1 = v1.norm(dim=1, keepdim=True) + 1e-9
            norm2 = v2.norm(dim=1, keepdim=True) + 1e-9
            u1 = v1 / norm1
            u2 = v2 / norm2
            dot = (u1 * u2).sum(dim=1, keepdim=True)
            Fi = -k_angle * dot * (u2 - dot * u1) / norm1
            Fj = -k_angle * dot * (u1 - dot * u2) / norm2
            F.index_add_(0, pair_n1, Fi)
            F.index_add_(0, pair_n2, Fj)
            F.index_add_(0, pair_hub, -(Fi + Fj))

        # repulsion
        if k_rep.item() > 0:
            diff_full = pos[:, None, :] - pos[None, :, :]
            dist2_full = (diff_full * diff_full).sum(dim=2) + 1e-6
            inv_dist = dist2_full.sqrt().reciprocal()
            rep_mag = k_rep * inv_dist * inv_dist
            force_full = diff_full * rep_mag[:, :, None] * inv_dist[:, :, None]
            force_full[torch.arange(pos.shape[0]), torch.arange(pos.shape[0])] = 0.0
            F = F + force_full.sum(dim=1)

        # simple Euler step
        acc = F  # mass=1
        vel = vel + dt * acc
        pos = pos + dt * vel

        # loss: weighted proximity for skip-gram pairs
        pi = pos[skip_src_t]
        pj = pos[skip_dst_t]
        dist_skip = torch.linalg.norm(pi - pj, dim=1)
        loss = (skip_w_t * dist_skip).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % max(1, iters // 5) == 0:
            print(f"[train-k] step {step+1}/{iters} loss={loss.item():.4f} ks={k_spring.item():.3f} kr={k_rep.item():.3f} ka={k_angle.item():.3f}")

    return {
        "k_spring": float(torch.nn.functional.softplus(k_spring_p).item()),
        "k_rep": float(torch.nn.functional.softplus(k_rep_p).item()),
        "k_angle": float(torch.nn.functional.softplus(k_angle_p).item()),
    }


def ensure_gutenberg():
    try:
        _ = gutenberg.fileids()
    except LookupError:
        nltk.download("gutenberg")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/words")
    except LookupError:
        nltk.download("words")


def pick_random_fileid() -> str:
    fileids = gutenberg.fileids()
    if not fileids:
        raise SystemExit("No Gutenberg texts available even after download.")
    return random.choice(fileids)


def slice_sentences(fileid: str, nlp, limit: int = 64, random_start: bool = False) -> str:
    # Materialize sentences to allow random start selection
    all_sents = list(gutenberg.sents(fileid))
    if not all_sents:
        raise SystemExit("No Gutenberg sentences available after download.")
    start = 0
    if random_start and len(all_sents) > limit:
        start = random.randint(0, max(0, len(all_sents) - limit))
    chunk = all_sents[start : start + limit]
    sentences = [" ".join(s) for s in chunk]
    return " ".join(sentences), start, len(all_sents)


def build_text_lexicon(text: str, nlp, max_words: int = 20000) -> list[dict[str, str]]:
    """Build lexicon from the source text itself (unique words, POS-tagged)."""
    tokens = re.findall(r"[A-Za-z]+", text)
    counts = Counter(w.lower() for w in tokens)
    vocab = [w for w, _ in counts.most_common(max_words)]
    lexicon: list[dict[str, str]] = []
    for doc in nlp.pipe(vocab, batch_size=512, disable=["ner", "parser"]):
        for token in doc:
            if not token.is_alpha:
                continue
            lexicon.append({"word": token.text, "pos": token.pos_.upper()})
    return lexicon


def _parse_kv_list(pairs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        try:
            out[key] = float(val)
        except ValueError:
            continue
    return out

    out: dict[str, float] = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        try:
            out[key] = float(val)
        except ValueError:
            continue
    return out


def _load_corpus_tokens(path: str) -> list[str]:
    tokens: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # allow space-separated tokens per line
            parts = line.split()
            tokens.extend(parts)
    return tokens


def _make_temp_fn(
    base: float,
    amp: float,
    omega: float,
    omega_fm: float,
    omega_season: float,
    phase: float,
    base_season_amp: float = 0.0,
    amp_season_delta: float = 0.0,
    season_phase: float = 0.0,
):
    """Build a temperature schedule with daily swing plus seasonal bias/amplitude modulation."""
    if amp == 0.0 and base == 0.0 and base_season_amp == 0.0 and amp_season_delta == 0.0:
        return None

    def temp_fn(t: float) -> float:
        # Daily phase with optional frequency modulation term retained for backwards compatibility.
        phase_term = omega * t + omega_fm * math.sin(omega_season * t) + phase
        # Seasonal offsets applied to both the base (bias) and the swing amplitude.
        seasonal = math.sin(omega_season * t + season_phase)
        base_mod = base + base_season_amp * seasonal
        amp_mod = amp + amp_season_delta * seasonal
        return max(0.0, base_mod + amp_mod * math.sin(phase_term))

    return temp_fn


def _make_extra_force_fn(charges: np.ndarray, masses_extra: np.ndarray, k_charge: float, k_grav: float, softening: float, force_cap: float):
    charges = charges.astype(np.float32)
    masses_extra = masses_extra.astype(np.float32)

    def extra_force(pos: np.ndarray, scratch: dict):
        N, D = pos.shape
        if N == 0:
            return None
        diff_full = pos[:, None, :] - pos[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff_full, diff_full) + softening
        inv_dist = np.sqrt(dist2)
        inv_dist = np.reciprocal(inv_dist, where=inv_dist > 0)
        inv_dist3 = inv_dist * inv_dist * inv_dist

        F = np.zeros_like(pos, dtype=np.float32)

        if k_charge != 0.0 and np.any(charges):
            qq = np.outer(charges, charges)
            np.fill_diagonal(qq, 0.0)
            force_mag = k_charge * qq * inv_dist3
            force_mag = np.nan_to_num(force_mag, nan=0.0, posinf=0.0, neginf=0.0)
            Fc = diff_full * force_mag[..., None]
            F += Fc.sum(axis=1)

        if k_grav != 0.0 and np.any(masses_extra):
            mm = np.outer(masses_extra, masses_extra)
            np.fill_diagonal(mm, 0.0)
            force_mag = -k_grav * mm * inv_dist3
            force_mag = np.nan_to_num(force_mag, nan=0.0, posinf=0.0, neginf=0.0)
            Fg = diff_full * force_mag[..., None]
            F += Fg.sum(axis=1)

        if force_cap and force_cap > 0:
            norms = np.linalg.norm(F, axis=1)
            over = norms > force_cap
            if np.any(over):
                scale = force_cap / (norms[over] + 1e-9)
                F[over] *= scale[:, None]
        return F

    return extra_force


def _make_extra_force_fn_torch(charges: np.ndarray, masses_extra: np.ndarray, k_charge: float, k_grav: float, softening: float, force_cap: float, device: str):
    if torch is None:
        return None
    charges_t = torch.tensor(charges, dtype=torch.float32, device=device)
    masses_t = torch.tensor(masses_extra, dtype=torch.float32, device=device)
    soft_t = torch.tensor(softening, dtype=torch.float32, device=device)
    cap_t = torch.tensor(force_cap, dtype=torch.float32, device=device)

    def extra_force_t(pos_t):
        N = pos_t.shape[0]
        if N == 0:
            return None
        diff = pos_t[:, None, :] - pos_t[None, :, :]
        dist2 = (diff * diff).sum(dim=2) + soft_t
        inv_r = dist2.sqrt().reciprocal()
        inv_r3 = inv_r * inv_r * inv_r
        F = torch.zeros_like(pos_t)
        if k_charge != 0.0 and torch.any(charges_t != 0):
            qq = torch.outer(charges_t, charges_t)
            qq.fill_diagonal_(0.0)
            fm = k_charge * qq * inv_r3
            F = F + (diff * fm.unsqueeze(-1)).sum(dim=1)
        if k_grav != 0.0 and torch.any(masses_t != 0):
            mm = torch.outer(masses_t, masses_t)
            mm.fill_diagonal_(0.0)
            fm = -k_grav * mm * inv_r3
            F = F + (diff * fm.unsqueeze(-1)).sum(dim=1)
        if force_cap and force_cap > 0:
            norms = torch.linalg.norm(F, dim=1)
            over = norms > cap_t
            if torch.any(over):
                scale = cap_t / (norms[over] + 1e-9)
                F[over] = F[over] * scale.unsqueeze(-1)
        return F

    return extra_force_t


def _make_dynamic_extra_force_fn(state, lock, k_charge: float, k_grav: float, softening: float, force_cap: float):
    soft = softening
    cap = force_cap

    def extra_force(pos: np.ndarray, scratch: dict):
        with lock:
            charges = state.get("charges")
            masses = state.get("masses")
        if charges is None or masses is None:
            return None
        N, D = pos.shape
        if N == 0:
            return None
        diff_full = pos[:, None, :] - pos[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff_full, diff_full) + soft
        inv_r = np.sqrt(dist2)
        inv_r = np.reciprocal(inv_r, where=inv_r > 0)
        inv_r3 = inv_r * inv_r * inv_r
        F = np.zeros_like(pos, dtype=np.float32)
        if k_charge != 0.0 and np.any(charges):
            qq = np.outer(charges, charges)
            np.fill_diagonal(qq, 0.0)
            fm = k_charge * qq * inv_r3
            fm = np.nan_to_num(fm, nan=0.0, posinf=0.0, neginf=0.0)
            F += (diff_full * fm[..., None]).sum(axis=1)
        if k_grav != 0.0 and np.any(masses):
            mm = np.outer(masses, masses)
            np.fill_diagonal(mm, 0.0)
            fm = -k_grav * mm * inv_r3
            fm = np.nan_to_num(fm, nan=0.0, posinf=0.0, neginf=0.0)
            F += (diff_full * fm[..., None]).sum(axis=1)
        if cap and cap > 0:
            norms = np.linalg.norm(F, axis=1)
            over = norms > cap
            if np.any(over):
                scale = cap / (norms[over] + 1e-9)
                F[over] *= scale[:, None]
        return F

    return extra_force


def _make_dynamic_extra_force_fn_torch(state, lock, k_charge: float, k_grav: float, softening: float, force_cap: float, device: str):
    if torch is None:
        return None
    soft_t = torch.tensor(softening, dtype=torch.float32, device=device)
    cap_t = torch.tensor(force_cap, dtype=torch.float32, device=device)

    def extra_force_t(pos_t):
        with lock:
            charges_t = state.get("charges_t")
            masses_t = state.get("masses_t")
        if charges_t is None or masses_t is None:
            return None
        if pos_t.numel() == 0:
            return None
        diff = pos_t[:, None, :] - pos_t[None, :, :]
        dist2 = (diff * diff).sum(dim=2) + soft_t
        inv_r = dist2.sqrt().reciprocal()
        inv_r3 = inv_r * inv_r * inv_r
        F = torch.zeros_like(pos_t)
        if k_charge != 0.0 and torch.any(charges_t != 0):
            qq = torch.outer(charges_t, charges_t)
            qq.fill_diagonal_(0.0)
            fm = k_charge * qq * inv_r3
            F = F + (diff * fm.unsqueeze(-1)).sum(dim=1)
        if k_grav != 0.0 and torch.any(masses_t != 0):
            mm = torch.outer(masses_t, masses_t)
            mm.fill_diagonal_(0.0)
            fm = -k_grav * mm * inv_r3
            F = F + (diff * fm.unsqueeze(-1)).sum(dim=1)
        if force_cap and force_cap > 0:
            norms = torch.linalg.norm(F, dim=1)
            over = norms > cap_t
            if torch.any(over):
                scale = cap_t / (norms[over] + 1e-9)
                F[over] = F[over] * scale.unsqueeze(-1)
        return F

    return extra_force_t


def main():
    parser = argparse.ArgumentParser(description="Run animator on 64 sentences from a random Gutenberg text")
    parser.add_argument("--limit", type=int, default=64, help="Number of sentences to keep")
    parser.add_argument("--fileid", type=str, default=None, help="Specific NLTK Gutenberg fileid to use (e.g., bible-kjv.txt)")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--k-spring", type=float, default=1.0)
    parser.add_argument("--k-rep", type=float, default=0.0)
    parser.add_argument("--k-angle", type=float, default=0.0, help="Angular spreading stiffness along edges")
    parser.add_argument("--temp", type=float, default=0.0, help="Thermal noise amplitude (set 0 to disable)")
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--softening", type=float, default=1e-2)
    parser.add_argument("--steps-per-frame", type=int, default=2)
    parser.add_argument("--max-speed", type=float, default=5.0, help="Clamp particle speeds (set 0 to disable)")
    parser.add_argument("--use-torch", action="store_true", help="Run physics on torch (GPU if available)")
    parser.add_argument("--torch-device", type=str, default="cuda", help="Torch device to use (cuda|cpu)")
    parser.add_argument("--train-k", action="store_true", help="Learn global k_spring/k_rep/k_angle from skip-gram proximity (torch only)")
    parser.add_argument("--train-k-live", action="store_true", help="Enable live k updates inside animator (torch only)")
    parser.add_argument("--train-k-iters", type=int, default=80, help="Iterations for k training loop")
    parser.add_argument("--train-k-lr", type=float, default=0.05, help="Learning rate for k training")
    parser.add_argument("--train-k-every", type=int, default=60, help="Frames between live k updates (set 1 for every frame)")
    parser.add_argument("--train-k-field", action="store_true", help="Predict ks with per-node/edge nets instead of globals (torch only)")
    parser.add_argument("--train-k-hidden", type=int, default=64, help="Hidden width for k field MLPs")
    parser.add_argument("--train-k-grad-clip", type=float, default=1.0, help="Gradient clip norm for live k training")
    parser.add_argument("--train-k-reg", type=float, default=0.0, help="L2 regularization weight for learned ks")
    parser.add_argument("--k-smooth", type=float, default=0.0, help="EMA smoothing factor (0-1) for k values")
    parser.add_argument("--k-norm-mean-floor", type=float, default=1e-3, help="If mean k drops below this, shift up to avoid collapse")
    parser.add_argument("--k-norm-sum1", action="store_true", help="Normalize each k set to sum to 1 after smoothing")
    parser.add_argument("--train-rest", action="store_true", help="Learn edge rest lengths during live training (torch only)")
    parser.add_argument("--rest-min", type=float, default=1e-4, help="Minimum rest length when learning rest (clamp)")
    parser.add_argument("--rest-max", type=float, default=None, help="Maximum rest length when learning rest (clamp; blank for none)")
    parser.add_argument("--use-global-ngrams", action="store_true", help="Augment training pairs with n-gram adjacencies from the full corpus text")
    parser.add_argument("--ngram-window", type=int, default=2, help="Window for global n-gram adjacency (tokens within this distance are positives)")
    parser.add_argument("--train-k-neg-samples", type=int, default=0, help="Number of negative pairs to sample per live train step (0 to disable)")
    parser.add_argument("--train-k-neg-margin", type=float, default=0.5, help="Hinge margin for negative pairs")
    parser.add_argument("--train-k-neg-weight", type=float, default=1.0, help="Loss weight for negative pairs")
    parser.add_argument("--train-k-thrust", action="store_true", help="Predict per-sentence thrust vectors from k field net (torch only)")
    parser.add_argument("--train-k-damp", action="store_true", help="Predict per-node damping from k field net (torch only)")
    parser.add_argument("--constrain-unit-sphere", action="store_true", help="Clamp nodes to unit sphere and remove radial velocity")
    parser.add_argument("--train-k-grad-acc", type=int, default=1, help="Accumulate gradients over this many live-train iterations before optimizer step")
    parser.add_argument("--train-k-node-depth", type=int, default=1, help="Hidden depth (>=1) for node k network trunk")
    parser.add_argument("--train-k-edge-depth", type=int, default=1, help="Hidden depth (>=1) for edge k network")
    parser.add_argument("--train-k-separate-heads", action="store_true", help="Use separate heads for rep/angle/damp/thrust (shared trunk)")
    parser.add_argument("--physics-lockstep", action="store_true", help="Run torch physics synchronously on the main thread (no background worker)")
    parser.add_argument("--train-k-live-lockstep", action="store_true", help="Run live k training each frame before physics step (requires --physics-lockstep)")
    parser.add_argument("--skipgram-window", type=int, default=2, help="Skip-gram window for co-occurrence weighting")
    parser.add_argument("--token-graph", action="store_true", help="Use token-based reversible graph instead of semantic graph")
    parser.add_argument("--loss-source-sent-idx", type=int, default=None, help="Token graph: sentence index in the selected chunk to use as loss source (0-based)")
    parser.add_argument("--loss-target-sent-idx", type=int, default=None, help="Token graph: sentence index in the selected chunk to align to (must have equal token length)")
    parser.add_argument(
        "--loss-target-auto-match",
        action="store_true",
        help="Token graph: pick the first other sentence in the chunk with equal token length as target (ignores --loss-target-sent-idx)",
    )
    parser.add_argument(
        "--loss-target-global-match",
        action="store_true",
        help="Token graph: pick the first sentence OUTSIDE the selected chunk with equal token length as target; adds that sentence/tokens as fixed nodes",
    )
    parser.add_argument("--show-k-colors", action="store_true", help="Visualize k rep/angle per node and k spring/rest per edge via colors")
    parser.add_argument("--log-embed-var", action="store_true", help="Log embedding variance/PCS for override graphs")
    parser.add_argument("--enable-regen", action="store_true", help="Enable background sentence regeneration from physics snapshots")
    parser.add_argument(
        "--regen-mode",
        choices=["deterministic", "gpt2", "bert"],
        default="gpt2",
        help="Regeneration mode: template clause, GPT-2, or BERT fill-mask",
    )
    parser.add_argument(
        "--regen-text",
        action="store_true",
        help="Also generate full sentences; by default only node labels update",
    )
    parser.add_argument(
        "--lexicon-source",
        choices=["text", "wordfreq"],
        default="text",
        help="Source for regen lexicon: unique words from the Gutenberg text (default) or wordfreq list",
    )
    parser.add_argument(
        "--max-lex-words",
        type=int,
        default=20000,
        help="Max vocabulary size for lexicon (applies to both sources)",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Pick a random start offset in the Gutenberg sentences before slicing the limit",
    )
    parser.add_argument(
        "--corpus",
        help="Path to corpus tokens/rules (one token or rule per line) for corpus-physics mode (disables LM regen)",
    )
    parser.add_argument(
        "--corpus-threshold",
        type=float,
        default=0.01,
        help="Cosine threshold to snap a node to a corpus token",
    )
    parser.add_argument(
        "--corpus-charge-k",
        type=float,
        default=0.5,
        help="Coulomb-like constant for charge forces in corpus mode",
    )
    parser.add_argument(
        "--corpus-grav-k",
        type=float,
        default=0.3,
        help="Gravity-like constant for mass attraction in corpus mode",
    )
    parser.add_argument(
        "--corpus-force-cap",
        type=float,
        default=5.0,
        help="Clamp magnitude of extra corpus forces per node (0 to disable)",
    )
    parser.add_argument(
        "--corpus-damping-map",
        nargs="*",
        default=["anchor=0.05"],
        help="Damping mapping entries token=value (extra damping per sub-step, space separated)",
    )
    parser.add_argument(
        "--corpus-inertia-map",
        nargs="*",
        default=["inertia=0.2"],
        help="Inertia entries token=value (fractional force scaling penalty 0..1)",
    )
    parser.add_argument(
        "--corpus-entropy-map",
        nargs="*",
        default=["entropy=0.02"],
        help="Entropy entries token=value (additive per-node temperature amplitude)",
    )
    parser.add_argument(
        "--corpus-spring-map",
        nargs="*",
        default=["spring=0.5"],
        help="Spring entries token=value (multiplier for edge stiffness when endpoint has token)",
    )
    parser.add_argument(
        "--corpus-charge-map",
        nargs="*",
        default=["e-=-1", "e+=1", "E-=-1", "E+=1", "charge=1"],
        help="Charge mapping entries token=value (space separated)",
    )
    parser.add_argument(
        "--corpus-mass-map",
        nargs="*",
        default=["gravity=1", "mass=1"],
        help="Mass mapping entries token=value (space separated); numeric tokens also accepted",
    )
    parser.add_argument("--temp-base", type=float, default=0.0, help="Base temperature for sinusoid schedule")
    parser.add_argument("--temp-amp", type=float, default=0.0, help="Amplitude for temperature sinusoid")
    parser.add_argument(
        "--temp-omega",
        type=float,
        default=0.5,
        help="Angular frequency (rad/s) for base temperature sinusoid",
    )
    parser.add_argument(
        "--temp-omega-fm",
        type=float,
        default=0.2,
        help="Frequency modulation depth (rad) applied to the base sinusoid phase",
    )
    parser.add_argument(
        "--temp-omega-season",
        type=float,
        default=0.05,
        help="Seasonal angular frequency (rad/s) used in FM term",
    )
    parser.add_argument("--temp-phase", type=float, default=0.0, help="Phase offset for temperature sinusoid")
    parser.add_argument(
        "--temp-season-base-amp",
        type=float,
        default=0.0,
        help="Seasonal sinusoid amplitude added to base temperature (bias shift)",
    )
    parser.add_argument(
        "--temp-season-amp-delta",
        type=float,
        default=0.0,
        help="Seasonal sinusoid amplitude added to the day/night swing amplitude",
    )
    parser.add_argument(
        "--temp-season-phase",
        type=float,
        default=0.0,
        help="Phase offset applied to the seasonal modulation (base and amplitude)",
    )
    args = parser.parse_args()

    ensure_gutenberg()
    if args.fileid:
        fid = args.fileid
        if fid not in gutenberg.fileids():
            raise SystemExit(f"Requested fileid '{fid}' not found in NLTK Gutenberg corpus. Available: {gutenberg.fileids()}")
    else:
        fid = pick_random_fileid()
    print(f"Loaded Gutenberg text: {fid}")
    all_sents = list(gutenberg.sents(fid))

    print("Loading models (spaCy + SBERT)...")
    nlp, sbert = load_models()

    print(f"Slicing {args.limit} sentences (random start={args.random_start})...")
    trimmed, start_idx, total_sents = slice_sentences(fid, nlp, limit=args.limit, random_start=args.random_start)
    print(f"Selected sentences {start_idx}..{start_idx + args.limit} of ~{total_sents}")

    if args.token_graph:
        print("Building token-based reversible graph...")
        nodes_override, edges_override = build_token_graph(trimmed, nlp, sbert, max_sentences=args.limit)
        if not nodes_override:
            raise SystemExit("Token graph construction yielded no nodes")

        if args.log_embed_var:
            mat = np.vstack([n.get("embedding", np.zeros(3, dtype=float)) for n in nodes_override])
            finite_mask = np.isfinite(mat).all(axis=1)
            mat = mat[finite_mask]
            if mat.size:
                mean = mat.mean(axis=0, keepdims=True)
                Xc = mat - mean
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                var = (S ** 2) / max(len(mat) - 1, 1)
                total = var.sum() + 1e-12
                ratios = var[:5] / total
                print(f"[embeddings] norms min/median/max: {np.linalg.norm(mat, axis=1).min():.4f} / {np.median(np.linalg.norm(mat, axis=1)):.4f} / {np.linalg.norm(mat, axis=1).max():.4f}")
                print(f"[embeddings] top PC variance ratios: {ratios.tolist()}")
            else:
                print("[embeddings] no finite embeddings to log")

        node_ids = [n["id"] for n in nodes_override]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        nodes_map = {n["id"]: n for n in nodes_override}

        # traversal order: sentence then its tokens
        sent_order = sorted([nid for nid in node_ids if nid.startswith("sent:")], key=lambda x: int(x.split(":")[1]))
        sent_proj = sorted([nid for nid in node_ids if nid.startswith("sentp:")], key=lambda x: int(x.split(":")[1]))
        token_order_by_sent = {}
        for sid in sent_order:
            parts = sid.split(":")
            si = int(parts[1]) if len(parts) > 1 else 0
            toks = [nid for nid in node_ids if nid.startswith(f"tok:{si}:")]
            toks = sorted(toks, key=lambda x: int(x.split(":")[-1]))
            token_order_by_sent[sid] = toks

        # flat token order for skip-grams
        flat_token_order = []
        for sid in sent_order:
            flat_token_order.extend(token_order_by_sent.get(sid, []))

        skip_src, skip_dst, skip_w = build_skipgram_pairs(flat_token_order, id_to_idx, window=args.skipgram_window)
        # local n-gram surface pairs (for exclusion from global)
        local_pairs: set[tuple[str, str]] = set()
        for sid in sent_order:
            toks = token_order_by_sent.get(sid, [])
            for a, src_nid in enumerate(toks):
                src_txt = nodes_map[src_nid].get("text")
                if not src_txt:
                    continue
                for off in range(1, args.ngram_window + 1):
                    b = a + off
                    if b >= len(toks):
                        break
                    dst_nid = toks[b]
                    dst_txt = nodes_map[dst_nid].get("text")
                    if not dst_txt:
                        continue
                    local_pairs.add((src_txt, dst_txt))

        # optional global n-gram positives from full corpus text (by surface form)
        if args.use_global_ngrams:
            full_tokens = list(gutenberg.words(fid))
            node_text_to_idx: dict[str, list[int]] = {}
            for nid, idx in id_to_idx.items():
                if nid.startswith("tok:"):
                    txt = nodes_map[nid].get("text")
                    if not txt:
                        continue
                    node_text_to_idx.setdefault(txt, []).append(idx)
            g_src, g_dst, g_w = build_global_ngram_pairs(
                full_tokens,
                node_text_to_idx,
                window=args.ngram_window,
                exclude_pairs=local_pairs,
            )
            if g_src:
                skip_src.extend(g_src)
                skip_dst.extend(g_dst)
                skip_w.extend(g_w)

        # optional sentence-to-sentence alignment loss (position-wise) within the selected chunk
        target_idx_resolved = None
        target_idx_resolved = None
        target_global_added_sid = None
        if args.loss_target_sent_idx is not None:
            target_idx_resolved = args.loss_target_sent_idx
        elif args.loss_target_auto_match:
            src_idx_candidate = args.loss_source_sent_idx if args.loss_source_sent_idx is not None else 0
            if src_idx_candidate < 0 or src_idx_candidate >= len(sent_order):
                raise SystemExit(f"--loss-source-sent-idx must be within [0, {len(sent_order) - 1}]")
            src_tokens_candidate = token_order_by_sent.get(sent_order[src_idx_candidate], [])
            if not src_tokens_candidate:
                raise SystemExit("Selected source sentence has no tokens to align")
            src_len = len(src_tokens_candidate)
            for idx_candidate, sid in enumerate(sent_order):
                if idx_candidate == src_idx_candidate:
                    continue
                cand_tokens = token_order_by_sent.get(sid, [])
                if len(cand_tokens) == src_len and cand_tokens:
                    target_idx_resolved = idx_candidate
                    break
            if target_idx_resolved is None:
                raise SystemExit("No other sentence in the chunk matches source token length for alignment")

        if target_idx_resolved is None and args.loss_target_global_match:
            src_idx_candidate = args.loss_source_sent_idx if args.loss_source_sent_idx is not None else 0
            if src_idx_candidate < 0 or src_idx_candidate >= len(sent_order):
                raise SystemExit(f"--loss-source-sent-idx must be within [0, {len(sent_order) - 1}]")
            src_tokens_candidate = token_order_by_sent.get(sent_order[src_idx_candidate], [])
            if not src_tokens_candidate:
                raise SystemExit("Selected source sentence has no tokens to align")
            src_len = len(src_tokens_candidate)
            chunk_lo = start_idx
            chunk_hi = start_idx + args.limit
            target_global_idx = None
            for gi, sent_tokens in enumerate(all_sents):
                if gi >= chunk_lo and gi < chunk_hi:
                    continue
                if len(sent_tokens) == src_len and len(sent_tokens) > 0:
                    target_global_idx = gi
                    target_global_tokens = [t for t in sent_tokens]
                    break
            if target_global_idx is None:
                raise SystemExit("No equal-length sentence outside the selected chunk was found for alignment")
            # Build fixed target sentence + tokens and add to graph
            target_sent_text = " ".join(target_global_tokens)
            sent_emb = sbert.encode([target_sent_text], normalize_embeddings=False)[0]
            tok_embs = sbert.encode(target_global_tokens, normalize_embeddings=False)
            tgt_sid = f"sentg:{target_global_idx}"
            nodes_override.append({"id": tgt_sid, "kind": "sentence", "text": target_sent_text, "embedding": sent_emb.tolist(), "fixed": True})
            target_token_ids: list[str] = []
            for ti, (tok_txt, tok_vec) in enumerate(zip(target_global_tokens, tok_embs)):
                tid = f"tokg:{target_global_idx}:{ti}"
                nodes_override.append({"id": tid, "kind": "token", "text": tok_txt, "ws": " ", "embedding": tok_vec.tolist(), "fixed": True})
                edges_override.append({"source": tgt_sid, "target": tid, "rel": "has_token", "k": 0.0, "physics": False})
                if ti > 0:
                    edges_override.append({"source": target_token_ids[-1], "target": tid, "rel": "next", "k": 0.0, "physics": False})
                target_token_ids.append(tid)
            # refresh node maps / orders with appended nodes
            for nid in [tgt_sid] + target_token_ids:
                node_ids.append(nid)
                id_to_idx[nid] = len(node_ids) - 1
                nodes_map[nid] = nodes_override[id_to_idx[nid]]
            sent_order.append(tgt_sid)
            token_order_by_sent[tgt_sid] = target_token_ids
            target_idx_resolved = len(sent_order) - 1
            target_global_added_sid = tgt_sid
            print(f"[loss-align] Added fixed target sentence {target_global_idx} from outside chunk with {len(target_token_ids)} tokens")

        if target_idx_resolved is not None:
            if target_idx_resolved < 0 or target_idx_resolved >= len(sent_order):
                raise SystemExit(f"Resolved target sentence idx must be within [0, {len(sent_order) - 1}]")
            src_idx = args.loss_source_sent_idx if args.loss_source_sent_idx is not None else 0
            if src_idx < 0 or src_idx >= len(sent_order):
                raise SystemExit(f"--loss-source-sent-idx must be within [0, {len(sent_order) - 1}]")
            src_tokens = token_order_by_sent.get(sent_order[src_idx], [])
            tgt_tokens = token_order_by_sent.get(sent_order[target_idx_resolved], [])
            if not src_tokens or not tgt_tokens:
                raise SystemExit("Selected source/target sentence has no tokens to align")
            if len(src_tokens) != len(tgt_tokens):
                raise SystemExit("Source and target sentences must have equal token length for alignment loss")
            skip_src = []
            skip_dst = []
            skip_w = []
            for pos in range(len(src_tokens)):
                si = id_to_idx[src_tokens[pos]]
                di = id_to_idx[tgt_tokens[pos]]
                skip_src.append(si)
                skip_dst.append(di)
                skip_w.append(1.0)
            print(
                f"[loss-align] Using position-aligned pairs between sentences {src_idx} and {target_idx_resolved} "
                f"({len(src_tokens)} tokens)"
            )

        learned_k = None
        if args.train_k:
            if torch is None:
                raise SystemExit("torch is required for --train-k")
            device = args.torch_device if torch.cuda.is_available() or args.torch_device == "cpu" else "cpu"
            learned_k = train_k_params(
                nodes_override,
                edges_override,
                skip_src,
                skip_dst,
                skip_w,
                device=device,
                iters=args.train_k_iters,
                lr=args.train_k_lr,
                dt=args.dt,
                k_spring_init=args.k_spring,
                k_rep_init=args.k_rep,
                k_angle_init=args.k_angle,
            )

        import threading, time

        highlight_state = {"current": set()}
        nodes_lock = threading.RLock()

        def recomposed_sentences():
            out = []
            with nodes_lock:
                for sid in sent_order:
                    toks = token_order_by_sent.get(sid, [])
                    parts = []
                    for tid in toks:
                        tnode = nodes_map.get(tid, {})
                        parts.append(tnode.get("text", ""))
                        parts.append(tnode.get("ws", ""))
                    out.append("".join(parts).strip())
            return out

        def highlight_provider():
            return list(highlight_state["current"])

        def label_provider():
            labels = []
            with nodes_lock:
                for idx in highlight_state["current"]:
                    if idx < 0 or idx >= len(node_ids):
                        continue
                    nid = node_ids[idx]
                    node = nodes_map.get(nid, {})
                    if nid.startswith("sent"):
                        si = int(nid.split(":")[1]) if ":" in nid else 0
                        recon = recomposed_sentences()
                        text = recon[si] if si < len(recon) else node.get("text", nid)
                    else:
                        text = node.get("text", nid)
                    labels.append({"idx": idx, "text": text[:96]})
            return labels

        def traversal_loop():
            while True:
                for sid in sent_order:
                    s_idx = id_to_idx.get(sid)
                    sp_idx = id_to_idx.get(f"sentp:{sid.split(':')[1]}") if ":" in sid else None
                    if s_idx is None:
                        continue
                    base_set = {s_idx}
                    if sp_idx is not None:
                        base_set.add(sp_idx)
                    highlight_state["current"] = base_set
                    time.sleep(0.7)
                    for tok_id in token_order_by_sent.get(sid, []):
                        t_idx = id_to_idx.get(tok_id)
                        if t_idx is None:
                            continue
                        highlight_state["current"] = base_set | {t_idx}
                        time.sleep(0.7)

        threading.Thread(target=traversal_loop, daemon=True).start()

        state_hook_fn = None
        if args.enable_regen:
            print("Token regen enabled; building lexicon for token substitution…")
            if args.lexicon_source == "text":
                raw_text = gutenberg.raw(fid)
                lexicon = build_text_lexicon(raw_text, nlp, max_words=args.max_lex_words)
            else:
                lexicon = load_wordfreq_lexicon(nlp, max_words=args.max_lex_words, batch_size=512)
            lex_words = [e["word"] for e in lexicon]
            lex_vecs = sbert.encode(lex_words, normalize_embeddings=True)
            print(f"Lexicon size for token regen: {len(lex_words)}")

            # optional GPU acceleration for regen similarity search
            use_torch_regen = bool(torch is not None and args.use_torch and torch.cuda.is_available())
            lex_vecs_t = None
            device_regen = args.torch_device if use_torch_regen else "cpu"
            if use_torch_regen:
                try:
                    lex_vecs_t = torch.tensor(lex_vecs, dtype=torch.float32, device=device_regen)
                    print(f"[regen] lexicon embeddings moved to {device_regen} for token regen")
                except Exception as exc:
                    print(f"[regen] warning: falling back to CPU for token regen ({exc})")
                    use_torch_regen = False

            snap_lock = threading.Lock()
            snap_state = {"pos": None, "ids": node_ids}
            first_snap_evt = threading.Event()

            def state_hook_fn(pos_arr, ids):
                if pos_arr is None:
                    return
                if not snap_lock.acquire(False):
                    return
                try:
                    snap_state["pos"] = np.array(pos_arr, copy=True)
                    snap_state["ids"] = list(ids)
                finally:
                    snap_lock.release()
                first_snap_evt.set()

            def regen_worker():
                import traceback

                first_snap_evt.wait()
                while True:
                    try:
                        first_snap_evt.wait(timeout=0.6)
                        with snap_lock:
                            pos = snap_state.get("pos")
                            ids = snap_state.get("ids", [])
                        if pos is None or pos.size == 0:
                            continue
                        id_to_idx_local = {nid: i for i, nid in enumerate(ids)}
                        tok_ids = [tid for tid in ids if tid.startswith("tok:")]
                        if not tok_ids:
                            continue
                        tok_idx = [id_to_idx_local[t] for t in tok_ids if t in id_to_idx_local]
                        if not tok_idx:
                            continue
                        pos_tok = pos[tok_idx]

                        batch = 4096
                        if use_torch_regen and lex_vecs_t is not None:
                            pos_t = torch.tensor(pos_tok, dtype=torch.float32, device=device_regen)
                            pos_t = pos_t / (pos_t.norm(dim=1, keepdim=True) + 1e-9)
                            best_idx = torch.zeros(pos_t.shape[0], dtype=torch.long, device=device_regen)
                            best_val = torch.full((pos_t.shape[0],), -1e9, device=device_regen)
                            for start in range(0, lex_vecs_t.shape[0], batch):
                                chunk = lex_vecs_t[start : start + batch]
                                sims_chunk = pos_t @ chunk.T
                                vals, idx = sims_chunk.max(dim=1)
                                better = vals > best_val
                                best_val = torch.where(better, vals, best_val)
                                best_idx = torch.where(better, idx + start, best_idx)
                            best_cpu = best_idx.cpu().numpy()
                        else:
                            best_cpu = None

                        if best_cpu is None:
                            pos_norm = pos_tok / (np.linalg.norm(pos_tok, axis=1, keepdims=True) + 1e-9)
                            best_list = []
                            for start in range(0, lex_vecs.shape[0], batch):
                                chunk = lex_vecs[start : start + batch]
                                sims = pos_norm @ chunk.T
                                if start == 0:
                                    best_val = sims.max(axis=1)
                                    best_idx = sims.argmax(axis=1) + start
                                else:
                                    val = sims.max(axis=1)
                                    idx = sims.argmax(axis=1) + start
                                    swap = val > best_val
                                    best_val = np.where(swap, val, best_val)
                                    best_idx = np.where(swap, idx, best_idx)
                            best_cpu = best_idx

                        with nodes_lock:
                            for tid, bi in zip(tok_ids, best_cpu.tolist()):
                                node = nodes_map.get(tid)
                                if node is None:
                                    continue
                                node["text"] = lex_words[bi]
                    except Exception as exc:
                        print(f"[regen][torch] exception: {exc}")
                        traceback.print_exc()
                        raise

            threading.Thread(target=regen_worker, daemon=True).start()

        print("Launching animator with token graph… Close the window to exit.")
        run_animator(
            path=None,
            dt=args.dt,
            k_spring=learned_k.get("k_spring", args.k_spring) if learned_k else args.k_spring,
            k_rep=learned_k.get("k_rep", args.k_rep) if learned_k else args.k_rep,
            k_angle=learned_k.get("k_angle", args.k_angle) if learned_k else args.k_angle,
            temp=args.temp,
            damping=args.damping,
            softening=args.softening,
            steps_per_frame=args.steps_per_frame,
            max_speed=None if args.max_speed == 0 else args.max_speed,
            use_torch=args.use_torch,
            torch_device=args.torch_device,
            nodes_override=nodes_override,
            edges_override=edges_override,
            highlight_provider=highlight_provider,
            label_provider=label_provider,
            state_hook=state_hook_fn,
            train_k_live=args.train_k_live,
            train_k_field=args.train_k_field,
            train_k_hidden=args.train_k_hidden,
            train_k_grad_clip=args.train_k_grad_clip,
            train_k_every=args.train_k_every,
            train_k_reg=args.train_k_reg,
            k_smooth=args.k_smooth,
            k_norm_mean_floor=args.k_norm_mean_floor,
            k_norm_sum1=args.k_norm_sum1,
            train_k_neg_samples=args.train_k_neg_samples,
            train_k_neg_margin=args.train_k_neg_margin,
            train_k_neg_weight=args.train_k_neg_weight,
            train_rest=args.train_rest,
            rest_min=args.rest_min,
            rest_max=args.rest_max,
            show_k_colors=args.show_k_colors,
            train_k_grad_acc=args.train_k_grad_acc,
            constrain_unit_sphere=args.constrain_unit_sphere,
            physics_lockstep=args.physics_lockstep,
            train_k_live_lockstep=args.train_k_live_lockstep,
            train_skip_src=np.array(skip_src, dtype=np.int64),
            train_skip_dst=np.array(skip_dst, dtype=np.int64),
            train_skip_w=np.array(skip_w, dtype=np.float32),
            train_k_lr=args.train_k_lr,
            train_k_unroll=1,
        )
        return

    print("Building graph + embeddings + edge distances...")
    G, node_vecs, edge_rows = graph_with_embeddings(trimmed, nlp, sbert, normalize=False)

    node_ids = [n for n, _ in G.nodes(data=True)]
    node_kinds_map = {n: d.get("kind", "concept") for n, d in G.nodes(data=True)}
    node_kinds = [node_kinds_map.get(n, "concept") for n in node_ids]
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    corpus_mode = args.corpus is not None
    corpus_labels: list[str] = [n for n in node_ids]
    extra_force_fn = None
    extra_force_fn_t = None
    trait_state = {
        "charges": None,
        "masses": None,
        "charges_t": None,
        "masses_t": None,
        "damping": None,
        "damping_t": None,
        "temp": None,
        "temp_t": None,
        "force_scale": None,
        "force_scale_t": None,
    }
    trait_lock = threading.Lock()
    charge_map = _parse_kv_list(args.corpus_charge_map)
    mass_map = _parse_kv_list(args.corpus_mass_map)
    damping_map = _parse_kv_list(args.corpus_damping_map)
    inertia_map = _parse_kv_list(args.corpus_inertia_map)
    entropy_map = _parse_kv_list(args.corpus_entropy_map)
    spring_map = _parse_kv_list(args.corpus_spring_map)
    with trait_lock:
        trait_state["charges"] = np.zeros(len(node_ids), dtype=np.float32)
        trait_state["masses"] = np.zeros(len(node_ids), dtype=np.float32)
        trait_state["damping"] = np.full(len(node_ids), args.damping, dtype=np.float32)
        trait_state["temp"] = np.full(len(node_ids), args.temp, dtype=np.float32)
        trait_state["force_scale"] = np.ones(len(node_ids), dtype=np.float32)
    torch_dev_pref = args.torch_device if (torch is not None and (torch.cuda.is_available() or args.torch_device == "cpu")) else "cpu"
    if args.use_torch and torch is not None:
        with trait_lock:
            trait_state["charges_t"] = torch.zeros(len(node_ids), dtype=torch.float32, device=torch_dev_pref)
            trait_state["masses_t"] = torch.zeros(len(node_ids), dtype=torch.float32, device=torch_dev_pref)
    temp_fn = _make_temp_fn(
        args.temp_base,
        args.temp_amp,
        args.temp_omega,
        args.temp_omega_fm,
        args.temp_omega_season,
        args.temp_phase,
        base_season_amp=args.temp_season_base_amp,
        amp_season_delta=args.temp_season_amp_delta,
        season_phase=args.temp_season_phase,
    )

    token_vecs = None
    tokens = None
    damping_vec = np.full(len(node_ids), args.damping, dtype=np.float32)
    temp_vec = np.full(len(node_ids), args.temp, dtype=np.float32)
    force_scale_vec = np.ones(len(node_ids), dtype=np.float32)
    node_spring_mul = np.ones(len(node_ids), dtype=np.float32)
    masses_vec = np.zeros(len(node_ids), dtype=np.float32)
    spring_state = {"scale": np.ones(len(edge_rows), dtype=np.float32)}
    spring_state_t = {"scale": None}

    if corpus_mode:
        print(f"Corpus mode enabled using: {args.corpus}")
        tokens = _load_corpus_tokens(args.corpus)
        if not tokens:
            raise SystemExit("Corpus file had no tokens")
        token_vecs = sbert.encode(tokens, normalize_embeddings=True)
        node_vec_mat = np.vstack([node_vecs[n] for n in node_ids])
        node_vec_norm = node_vec_mat / (np.linalg.norm(node_vec_mat, axis=1, keepdims=True) + 1e-9)
        sims = node_vec_norm @ token_vecs.T
        best_idx = np.argmax(sims, axis=1)
        best_scores = sims[np.arange(len(node_ids)), best_idx]
        charges = np.zeros(len(node_ids), dtype=np.float32)
        for i, nid in enumerate(node_ids):
            score = best_scores[i]
            tok = tokens[best_idx[i]] if score >= args.corpus_threshold else None
            if tok is None:
                corpus_labels[i] = nid
                continue
            corpus_labels[i] = tok
            charges[i] = charge_map.get(tok, 0.0)
            if tok in mass_map:
                masses_vec[i] = mass_map[tok]
            else:
                try:
                    masses_vec[i] = float(tok)
                except ValueError:
                    masses_vec[i] = 0.0
            if tok in damping_map:
                damping_vec[i] = min(0.99, max(0.0, args.damping + damping_map[tok]))
            if tok in inertia_map:
                penalty = max(0.0, min(1.0, inertia_map[tok]))
                force_scale_vec[i] = max(0.0, 1.0 - penalty)
                damping_vec[i] = min(0.99, damping_vec[i] + penalty)
            if tok in entropy_map:
                temp_vec[i] = max(0.0, temp_vec[i] + entropy_map[tok])
            if tok in spring_map:
                node_spring_mul[i] = min(node_spring_mul[i], max(0.0, spring_map[tok]))
        with trait_lock:
            trait_state["charges"] = charges.astype(np.float32)
            trait_state["masses"] = masses_vec.astype(np.float32)
            trait_state["damping"] = damping_vec.astype(np.float32)
            trait_state["temp"] = temp_vec.astype(np.float32)
            trait_state["force_scale"] = force_scale_vec.astype(np.float32)
        extra_force_fn = _make_dynamic_extra_force_fn(
            trait_state,
            trait_lock,
            args.corpus_charge_k,
            args.corpus_grav_k,
            args.softening,
            args.corpus_force_cap,
        )
        if args.use_torch and torch is not None:
            device = args.torch_device if (torch.cuda.is_available() or args.torch_device == "cpu") else "cpu"
            with trait_lock:
                trait_state["charges_t"] = torch.tensor(trait_state["charges"], dtype=torch.float32, device=device)
                trait_state["masses_t"] = torch.tensor(trait_state["masses"], dtype=torch.float32, device=device)
                trait_state["damping_t"] = torch.tensor(trait_state["damping"], dtype=torch.float32, device=device)
                trait_state["temp_t"] = torch.tensor(trait_state["temp"], dtype=torch.float32, device=device)
                trait_state["force_scale_t"] = torch.tensor(trait_state["force_scale"], dtype=torch.float32, device=device)
                spring_state_t["scale"] = torch.tensor(spring_state["scale"], dtype=torch.float32, device=device)
            extra_force_fn_t = _make_dynamic_extra_force_fn_torch(
                trait_state,
                trait_lock,
                args.corpus_charge_k,
                args.corpus_grav_k,
                args.softening,
                args.corpus_force_cap,
                device,
            )
        # update label overlay with snapped tokens
        node_labels_state = {"labels": [{"idx": i, "text": corpus_labels[i]} for i in range(len(node_ids))]}

    # optional background regeneration setup (opt-in via flag)
    snap_lock = threading.Lock()
    snap_state = {"pos": None, "ids": node_ids}
    snap_stats = {"snap_accepted": 0, "snap_dropped": 0, "regen_runs": 0}
    text_lines_state = {"lines": []}
    if not corpus_mode:
        node_labels_state = {"labels": [{"idx": i, "text": node_ids[i]} for i in range(len(node_ids))]}
        with trait_lock:
            trait_state["charges"] = np.zeros(len(node_ids), dtype=np.float32)
            trait_state["masses"] = masses_vec
            trait_state["damping"] = damping_vec
            trait_state["temp"] = temp_vec
            trait_state["force_scale"] = force_scale_vec
            if args.use_torch and torch is not None:
                device = args.torch_device if (torch.cuda.is_available() or args.torch_device == "cpu") else "cpu"
                trait_state["charges_t"] = torch.tensor(trait_state["charges"], dtype=torch.float32, device=device)
                trait_state["masses_t"] = torch.tensor(trait_state["masses"], dtype=torch.float32, device=device)
                trait_state["damping_t"] = torch.tensor(trait_state["damping"], dtype=torch.float32, device=device)
                trait_state["temp_t"] = torch.tensor(trait_state["temp"], dtype=torch.float32, device=device)
                trait_state["force_scale_t"] = torch.tensor(trait_state["force_scale"], dtype=torch.float32, device=device)
                spring_state_t["scale"] = torch.tensor(spring_state["scale"], dtype=torch.float32, device=device)
    stop_evt = threading.Event()
    first_snap_evt = threading.Event()
    worker = None
    state_hook = None

    if args.enable_regen:
        lexicon = None
        lex_vecs = None
        if not corpus_mode:
            if args.lexicon_source == "text":
                print("Building lexicon from source text vocabulary...")
                raw_text = gutenberg.raw(fid)
                lexicon = build_text_lexicon(raw_text, nlp, max_words=args.max_lex_words)
            else:
                print("Building pruned lexicon (wordfreq + NLTK words)...")
                lexicon = load_wordfreq_lexicon(nlp, max_words=args.max_lex_words, batch_size=512)
            print(f"Lexicon size: {len(lexicon)}")
            print("Encoding lexicon embeddings once for regen...")
            lex_words = [e["word"] for e in lexicon]
            lex_vecs = sbert.encode(lex_words, normalize_embeddings=True)
            print("Lexicon embeddings cached. Regen worker will use them for generation.")

        gen_device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        mode = args.regen_mode
        print(
            f"Regen enabled; mode={mode}; device={gen_device}; "
            f"sentences={'on' if args.regen_text else 'off'}; waiting for first physics snapshot..."
        )
        tok_model = {"tok": None, "model": None}
        bert_model = {"tok": None, "model": None}

        def state_hook(pos_arr, ids):
            if pos_arr is None:
                return
            # non-blocking snapshot to avoid stalling render/physics; drop if busy
            acquired = snap_lock.acquire(blocking=False)
            if not acquired:
                snap_stats["snap_dropped"] += 1
                return
            try:
                snap_state["pos"] = np.array(pos_arr, copy=True)
                snap_state["ids"] = list(ids)
                snap_stats["snap_accepted"] += 1
                if snap_stats["snap_accepted"] == 1:
                    print("[regen] first snapshot captured; background worker will start soon")
                elif snap_stats["snap_accepted"] % 25 == 0:
                    print(f"[regen] snapshots accepted={snap_stats['snap_accepted']} dropped={snap_stats['snap_dropped']}")
            finally:
                snap_lock.release()
            first_snap_evt.set()

        def bg_worker():
            # wait until we have at least one snapshot so we don't spin early
            first_snap_evt.wait()
            print("[regen] worker awake; waiting on snapshots to generate text")
            while not stop_evt.is_set():
                # wait with timeout to allow clean exit
                first_snap_evt.wait(timeout=3.0)
                with snap_lock:
                    pos = snap_state.get("pos")
                    ids = snap_state.get("ids", [])
                    accepted = snap_stats["snap_accepted"]
                    dropped = snap_stats["snap_dropped"]
                if pos is None or pos.size == 0:
                    print("[regen] no snapshot yet; sleeping")
                    continue
                kinds = [node_kinds_map.get(n, "concept") for n in ids]
                try:
                    print(f"[regen] snapshot ready: nodes={len(ids)} pos_shape={pos.shape}")
                    if corpus_mode:
                        pos_norm = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-9)
                        sims = pos_norm @ token_vecs.T
                        best_idx = np.argmax(sims, axis=1)
                        choices = {}
                        for i, nid in enumerate(ids):
                            choices[nid] = [(tokens[best_idx[i]], "", float(sims[i, best_idx[i]]))]
                    else:
                        choices = suggest_simple_words_for_nodes(
                            ids,
                            kinds,
                            pos,
                            sbert,
                            lexicon=lexicon,
                            lexicon_embeddings=lex_vecs,
                            top_k=2,
                        )
                        empty_nodes = [k for k, v in choices.items() if not v]
                        if empty_nodes:
                            print(f"[regen] warning: {len(empty_nodes)} nodes had no lexical choices (pos filter)")
                        print(f"[regen] lexical choices computed for {len(choices)} nodes")

                    sentences: list[str] = []
                    # update traits dynamically from current top labels
                    labels_for_traits: list[str] = []
                    for nid in ids:
                        top = choices.get(nid, [])
                        labels_for_traits.append(top[0][0] if top else str(nid))
                    charges = np.zeros(len(ids), dtype=np.float32)
                    masses_dyn = np.zeros(len(ids), dtype=np.float32)
                    damping_dyn = np.full(len(ids), args.damping, dtype=np.float32)
                    temp_dyn = np.full(len(ids), args.temp, dtype=np.float32)
                    force_scale_dyn = np.ones(len(ids), dtype=np.float32)
                    node_spring_dyn = np.ones(len(ids), dtype=np.float32)
                    for i, lbl in enumerate(labels_for_traits):
                        charges[i] = charge_map.get(lbl, 0.0)
                        if lbl in mass_map:
                            masses_dyn[i] = mass_map[lbl]
                        else:
                            try:
                                masses_dyn[i] = float(lbl)
                            except ValueError:
                                masses_dyn[i] = 0.0
                        if lbl in damping_map:
                            damping_dyn[i] = min(0.99, max(0.0, args.damping + damping_map[lbl]))
                        if lbl in inertia_map:
                            penalty = max(0.0, min(1.0, inertia_map[lbl]))
                            force_scale_dyn[i] = max(0.0, 1.0 - penalty)
                            damping_dyn[i] = min(0.99, damping_dyn[i] + penalty)
                        if lbl in entropy_map:
                            temp_dyn[i] = max(0.0, temp_dyn[i] + entropy_map[lbl])
                        if lbl in spring_map:
                            node_spring_dyn[i] = min(node_spring_dyn[i], max(0.0, spring_map[lbl]))
                    spring_scale_dyn = compute_spring_scale(node_spring_dyn)
                    with trait_lock:
                        trait_state["charges"] = charges
                        trait_state["masses"] = masses_dyn
                        trait_state["damping"] = damping_dyn
                        trait_state["temp"] = temp_dyn
                        trait_state["force_scale"] = force_scale_dyn
                        spring_state["scale"] = spring_scale_dyn
                        if args.use_torch and torch is not None:
                            device_t = gen_device
                            trait_state["charges_t"] = torch.tensor(charges, dtype=torch.float32, device=device_t)
                            trait_state["masses_t"] = torch.tensor(masses_dyn, dtype=torch.float32, device=device_t)
                            trait_state["damping_t"] = torch.tensor(damping_dyn, dtype=torch.float32, device=device_t)
                            trait_state["temp_t"] = torch.tensor(temp_dyn, dtype=torch.float32, device=device_t)
                            trait_state["force_scale_t"] = torch.tensor(force_scale_dyn, dtype=torch.float32, device=device_t)
                            spring_state_t["scale"] = torch.tensor(spring_scale_dyn, dtype=torch.float32, device=device_t)

                    if args.regen_text:
                        if mode == "deterministic":
                            print("[regen] generating deterministic sentences (no LM)...")
                            sentences = deterministic_sentences_from_graph(G, choices)
                        elif mode == "bert":
                            if bert_model["tok"] is None or bert_model["model"] is None:
                                tok, mdl = load_bert_fill_mask(device=gen_device)
                                bert_model["tok"], bert_model["model"] = tok, mdl
                                print("[regen] BERT fill-mask loaded; starting generation")
                            print("[regen] generating sentences with BERT fill-mask...")
                            sentences = fill_mask_sentences_from_graph(
                                G,
                                choices,
                                bert_model["tok"],
                                bert_model["model"],
                                mask_lengths=(1, 2, 3),
                                max_candidates=32,
                                device=gen_device,
                            )
                        else:
                            if tok_model["tok"] is None or tok_model["model"] is None:
                                tok, mdl = load_gpt2_small(device=gen_device)
                                tok_model["tok"], tok_model["model"] = tok, mdl
                                print("[regen] GPT-2 loaded; starting generation")
                            print("[regen] generating sentences with GPT-2...")
                            sentences = generate_sentences_from_graph(
                                G,
                                choices,
                                tok_model["tok"],
                                tok_model["model"],
                                beam_size=5,
                                max_new_tokens=40,
                                device=gen_device,
                            )
                        snap_stats["regen_runs"] += 1
                        lines = [f"gen[{i}]: {s}" for i, s in enumerate(sentences)]
                        text_lines_state["lines"] = lines
                    else:
                        text_lines_state["lines"] = []

                    # update per-node labels from choices (top word) or fallback to id
                    lbls = []
                    for idx, nid in enumerate(ids):
                        top = choices.get(nid, [])
                        word = top[0][0] if top else str(nid)
                        lbls.append({"idx": idx, "text": word})
                    node_labels_state["labels"] = lbls
                    print(
                        f"\n[regen] run={snap_stats['regen_runs']} snaps(acc/dropped)={accepted}/{dropped} nodes={len(ids)}"
                    )
                    if args.regen_text:
                        for idx, s in enumerate(sentences[:5]):
                            print(f"  - gen[{idx}]: {s}")
                        print("[regen] ...\n")
                except Exception as exc:
                    print(f"[regen warn] generation failed: {exc}")

        worker = threading.Thread(target=bg_worker, daemon=True)
        worker.start()

    # Build a pre-physics-like structure for animator
    nodes = []
    for nid, attrs in G.nodes(data=True):
        node = {
            "id": nid,
            "kind": attrs.get("kind", "concept"),
            "embedding": node_vecs[nid].tolist(),
        }
        node_idx_val = node_idx.get(nid)
        if node_idx_val is not None:
            node["mass"] = float(1.0 + masses_vec[node_idx_val])
        if "text" in attrs:
            node["text"] = attrs["text"]
        if "label" in attrs:
            node["lemma"] = attrs["label"]
        if "stmt_type" in attrs:
            node["stmt_type"] = attrs["stmt_type"]
        if "stmt_conf" in attrs:
            node["stmt_conf"] = float(attrs["stmt_conf"])
        nodes.append(node)

    edges = edge_rows  # already has source, target, distance, rel, rel_type

    def compute_spring_scale(node_factors: np.ndarray) -> np.ndarray:
        scale = np.ones(len(edges), dtype=np.float32)
        for idx, e in enumerate(edges):
            src = e.get("source")
            tgt = e.get("target")
            if src in node_idx and tgt in node_idx:
                a = node_factors[node_idx[src]]
                b = node_factors[node_idx[tgt]]
                scale[idx] = min(a, b)
        return scale

    spring_scale_vec = compute_spring_scale(node_spring_mul)
    spring_state = {"scale": spring_scale_vec}
    spring_state_t = {"scale": None}

    use_torch_flag = args.use_torch
    if extra_force_fn is None:
        extra_force_fn = _make_dynamic_extra_force_fn(
            trait_state,
            trait_lock,
            args.corpus_charge_k,
            args.corpus_grav_k,
            args.softening,
            args.corpus_force_cap,
        )
    if use_torch_flag and extra_force_fn_t is None and torch is not None:
        extra_force_fn_t = _make_dynamic_extra_force_fn_torch(
            trait_state,
            trait_lock,
            args.corpus_charge_k,
            args.corpus_grav_k,
            args.softening,
            args.corpus_force_cap,
            torch_dev_pref,
        )
    print("Launching animator...")
    run_animator(
        path=None,  # unused in this direct-call mode
        dt=args.dt,
        k_spring=args.k_spring,
        k_rep=args.k_rep,
        k_angle=args.k_angle,
        temp=args.temp,
        damping=args.damping,
        softening=args.softening,
        steps_per_frame=args.steps_per_frame,
        max_speed=None if args.max_speed == 0 else args.max_speed,
        use_torch=use_torch_flag,
        torch_device=args.torch_device,
        nodes_override=nodes,
        edges_override=edges,
        state_hook=state_hook,
        hook_interval_ms=1000,
        text_provider=lambda: text_lines_state["lines"],
        label_provider=lambda: node_labels_state["labels"],
        temp_fn=temp_fn,
        extra_force_fn=extra_force_fn,
        extra_force_fn_t=extra_force_fn_t,
        damping_per_node=damping_vec,
        damping_provider=lambda: trait_state.get("damping"),
        damping_provider_t=(lambda: trait_state.get("damping_t")) if args.use_torch and torch is not None else None,
        temp_per_node=temp_vec,
        temp_provider=lambda: trait_state.get("temp"),
        temp_provider_t=(lambda: trait_state.get("temp_t")) if args.use_torch and torch is not None else None,
        force_scale_per_node=force_scale_vec,
        force_scale_provider=lambda: trait_state.get("force_scale"),
        force_scale_provider_t=(lambda: trait_state.get("force_scale_t")) if args.use_torch and torch is not None else None,
        spring_k_scale=spring_state["scale"],
        spring_k_scale_provider=lambda: spring_state.get("scale"),
        spring_k_scale_provider_t=(lambda: spring_state_t.get("scale")) if args.use_torch and torch is not None else None,
        train_k_live=args.train_k_live,
        train_k_every=args.train_k_every,
        train_k_field=args.train_k_field,
        train_k_hidden=args.train_k_hidden,
        train_k_grad_clip=args.train_k_grad_clip,
        train_k_reg=args.train_k_reg,
        k_smooth=args.k_smooth,
        k_norm_mean_floor=args.k_norm_mean_floor,
        k_norm_sum1=args.k_norm_sum1,
        train_k_neg_samples=args.train_k_neg_samples,
        train_k_neg_margin=args.train_k_neg_margin,
        train_k_neg_weight=args.train_k_neg_weight,
        train_rest=args.train_rest,
        rest_min=args.rest_min,
        rest_max=args.rest_max,
        show_k_colors=args.show_k_colors,
        train_k_grad_acc=args.train_k_grad_acc,
        constrain_unit_sphere=args.constrain_unit_sphere,
        physics_lockstep=args.physics_lockstep,
        train_k_live_lockstep=args.train_k_live_lockstep,
        train_k_lr=args.train_k_lr,
    )

    # signal background worker to stop and exit
    if worker is not None:
        stop_evt.set()
        worker.join(timeout=1.0)


if __name__ == "__main__":
    main()
