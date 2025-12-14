"""
Lexicon-based word suggestion and GPT-2 sentence regeneration.

This module provides two main entry points:
- suggest_simple_words_for_nodes: pick nearest simple-lexicon words per node embedding.
- generate_sentences_from_graph: prompt GPT-2 with templated role hints to regenerate sentences.

Both are intentionally lightweight: you can pass your own lexicon and GPT-2 model/tokenizer.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple
import itertools

import numpy as np

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    GPT2LMHeadModel = None
    GPT2Tokenizer = None
    AutoModelForMaskedLM = None
    AutoTokenizer = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from wordfreq import top_n_list
except Exception as exc:  # pragma: no cover - required for lexicon build
    top_n_list = None
    _wordfreq_import_error = exc
else:
    _wordfreq_import_error = None

try:
    from nltk.corpus import words as nltk_words
except Exception:  # pragma: no cover - optional dependency
    nltk_words = None

DEFAULT_ALLOWED_POS_BY_KIND = {
    "concept": {"NOUN", "PROPN"},
    "modifier": {"ADJ", "ADV"},
    "verb": {"VERB"},
    "sentence": {"NOUN", "PROPN"},
}


def _simple_sentence_from_roles(roles: List[Tuple[str, str, str]]) -> str:
    """
    Build a plain sentence from role hints when the LM output is empty/echoes the prompt.

    Very basic: prefer sentence text if provided; otherwise join top words.
    """
    if not roles:
        return ""
    # if a sentence role exists, prefer its word directly
    for kind, word, _ in roles:
        if kind == "sentence" and word:
            return word.strip()
    words = [w for _, w, _ in roles if w]
    if not words:
        return ""
    sent = " ".join(words)
    # capitalize and add period if missing
    sent = sent[0:1].upper() + sent[1:]
    if not sent.endswith(('.', '!', '?')):
        sent += "."
    return sent


def _deterministic_clause(roles: List[Tuple[str, str, str]]) -> str:
    """Construct a simple clause from role words (subject/verb/object/modifiers)."""
    if not roles:
        return ""

    # Separate by kind for light structure
    sentence_words = [w for k, w, _ in roles if k == "sentence" and w]
    verbs = [w for k, w, _ in roles if k == "verb" and w]
    concepts = [w for k, w, _ in roles if k == "concept" and w]
    modifiers = [w for k, w, _ in roles if k == "modifier" and w]

    # Subject preference: explicit sentence word, else first concept
    subj = sentence_words[0] if sentence_words else (concepts[0] if concepts else "")
    # Verb preference: first verb
    verb = verbs[0] if verbs else ""
    # Object/compliment: next concept if available
    obj = concepts[1] if len(concepts) > 1 else ""
    # Build remainder from remaining concepts/modifiers
    rest = concepts[2:] if len(concepts) > 2 else []
    rest += modifiers

    tokens = [t for t in [subj, verb, obj] if t]
    tokens += rest
    if not tokens and roles:
        tokens = [w for _, w, _ in roles if w]
    if not tokens:
        return ""

    sent = " ".join(tokens)
    sent = sent[0:1].upper() + sent[1:]
    if not sent.endswith(('.', '!', '?')):
        sent += "."
    return sent


def _roles_for_sentence_node(G, sid: str, node_to_choices: Dict[str, List[Tuple[str, str, float]]]) -> List[Tuple[str, str, str]]:
    """Collect (kind, word, pos) tuples for successors of a sentence node."""
    roles: List[Tuple[str, str, str]] = []
    for nbr in G.successors(sid):
        if nbr in node_to_choices and node_to_choices[nbr]:
            word, pos, _ = node_to_choices[nbr][0]
            kind = G.nodes[nbr].get("kind", "concept")
            roles.append((kind, word, pos))
    if not roles:
        sent_text = G.nodes[sid].get("text", "")
        if sent_text:
            roles = [("sentence", sent_text, "")]
    return roles


def deterministic_sentences_from_graph(
    G,
    node_to_choices: Dict[str, List[Tuple[str, str, float]]],
) -> List[str]:
    """Build deterministic sentences from top role words (no LM)."""
    sent_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "sentence"]

    def _sent_key(sid: str):
        try:
            return int(sid.split(":")[1])
        except Exception:
            return sid

    sent_nodes = sorted(sent_nodes, key=_sent_key)
    outputs: List[str] = []

    for sid in sent_nodes:
        roles = []
        for nbr in G.successors(sid):
            if nbr in node_to_choices and node_to_choices[nbr]:
                word, pos, _ = node_to_choices[nbr][0]
                kind = G.nodes[nbr].get("kind", "concept")
                roles.append((kind, word, pos))
        if not roles:
            sent_text = G.nodes[sid].get("text", "")
            if sent_text:
                outputs.append(sent_text)
            else:
                outputs.append("")
            continue
        outputs.append(_deterministic_clause(roles))

    return outputs


def _load_dictionary_words() -> Set[str]:
    """Load an English wordlist (NLTK words corpus) to prune noisy tokens."""
    if nltk_words is None:
        raise ImportError("nltk words corpus is required to prune the lexicon; install nltk data")
    try:
        return {w.lower() for w in nltk_words.words()}
    except LookupError as exc:  # auto-download if present but missing
        try:
            import nltk

            nltk.download("words")
            return {w.lower() for w in nltk_words.words()}
        except Exception as inner:
            raise ImportError("nltk words corpus missing; run nltk.download('words')") from inner
    except Exception as exc:
        raise ImportError("failed to load nltk words corpus") from exc


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / norms


def load_wordfreq_lexicon(
    nlp,
    max_words: int = 20000,
    batch_size: int = 512,
    dictionary_words: Set[str] | None = None,
) -> List[Dict[str, str]]:
    """
    Build a lexicon from wordfreq's top-n list, POS-tagged by spaCy, pruned to dictionary words.

    Parameters
    ----------
    nlp : spaCy Language
        Loaded spaCy pipeline used for POS tagging.
    max_words : int
        Number of top words to include from wordfreq.
    batch_size : int
        spaCy pipe batch size for tagging.

    Parameters
    ----------
    dictionary_words : set[str] or None
        If provided, only keep words that appear in this dictionary (lowercased). Defaults to
        the NLTK words corpus; raises if that corpus is unavailable.

    Returns
    -------
    list of {"word": str, "pos": str}
    """
    if top_n_list is None:
        raise ImportError(f"wordfreq is required to build lexicon: {_wordfreq_import_error}")
    if nlp is None:
        raise ValueError("spaCy nlp pipeline is required to tag lexicon")
    if dictionary_words is None:
        dictionary_words = _load_dictionary_words()

    vocab = top_n_list("en", n=max_words, wordlist="best")
    lexicon: List[Dict[str, str]] = []
    for doc in nlp.pipe(vocab, batch_size=batch_size, disable=["ner", "parser"]):
        for token in doc:
            if not token.is_alpha:
                continue
            # prune to dictionary to avoid noisy web tokens
            if dictionary_words and token.text.lower() not in dictionary_words:
                continue
            pos = token.pos_.upper()
            lexicon.append({"word": token.text, "pos": pos})
    return lexicon


def suggest_simple_words_for_nodes(
    node_ids: Sequence[str],
    node_kinds: Sequence[str],
    node_embeddings: np.ndarray,
    embedder,
    lexicon: Sequence[Dict[str, str]] | None = None,
    lexicon_embeddings: np.ndarray | None = None,
    allowed_pos_by_kind: Dict[str, set[str]] | None = None,
    top_k: int = 3,
    min_score: float = 0.20,
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    For each node, pick top-k lexicon words by cosine similarity, constrained by POS.

    Parameters
    ----------
    node_ids : list[str]
    node_kinds : list[str]
    node_embeddings : np.ndarray (N,D)
    embedder : SentenceTransformer-like with .encode(texts, normalize_embeddings=True)
    lexicon : list of {"word": str, "pos": str}
    lexicon_embeddings : np.ndarray (L, D) optional
        Precomputed normalized embeddings for the lexicon words; if provided, avoids re-encoding
        the lexicon on every call and speeds up regen.
    allowed_pos_by_kind : mapping node_kind -> set of POS tags to allow
    top_k : number of candidates per node
    min_score : drop candidates below this cosine score

    Returns
    -------
    dict[node_id] -> list[(word, pos, score)] sorted by score desc
    """
    if lexicon is None:
        raise ValueError("lexicon is required; build via load_wordfreq_lexicon or provide your own")
    if allowed_pos_by_kind is None:
        allowed_pos_by_kind = DEFAULT_ALLOWED_POS_BY_KIND

    words = [e["word"] for e in lexicon]
    pos_tags = [e.get("pos", "").upper() for e in lexicon]
    if not words:
        return {}

    if lexicon_embeddings is None:
        lex_vecs = embedder.encode(words, normalize_embeddings=True)
    else:
        lex_vecs = lexicon_embeddings
    node_vecs = _normalize_rows(np.asarray(node_embeddings, dtype=np.float32))

    sims = node_vecs @ lex_vecs.T  # (N, L)
    result: Dict[str, List[Tuple[str, str, float]]] = {}
    for idx, nid in enumerate(node_ids):
        kind = node_kinds[idx] if idx < len(node_kinds) else "concept"
        allowed = allowed_pos_by_kind.get(kind, set())
        # build mask for allowed POS
        mask = np.array([p in allowed if allowed else True for p in pos_tags])
        if not mask.any():
            continue
        scores = sims[idx]
        scores = np.where(mask, scores, -1.0)
        top_idx = np.argsort(-scores)[: top_k + 2]  # a few extra in case of low scores
        picks: List[Tuple[str, str, float]] = []
        for j in top_idx:
            sc = float(scores[j])
            if sc < min_score:
                continue
            picks.append((words[j], pos_tags[j], sc))
            if len(picks) >= top_k:
                break
        result[nid] = picks
    return result


def load_gpt2_small(device: str = "cpu"):
    """Helper to load GPT-2 small with tokenizer on a chosen device."""
    if GPT2LMHeadModel is None or GPT2Tokenizer is None:
        raise ImportError("transformers is required for GPT-2 generation")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.padding_side = "left"  # needed for decoder-only models when padding batches
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    return tok, model


def load_bert_fill_mask(model_name: str = "bert-base-uncased", device: str = "cpu"):
    """Load a masked-LM model/tokenizer for fill-mask generation."""
    if AutoModelForMaskedLM is None or AutoTokenizer is None:
        raise ImportError("transformers is required for BERT fill-mask generation")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)
    mdl.to(device)
    return tok, mdl


def generate_sentences_from_graph(
    G,
    node_to_choices: Dict[str, List[Tuple[str, str, float]]],
    tokenizer,
    model,
    beam_size: int = 5,
    max_new_tokens: int = 40,
    device: str = "cpu",
) -> List[str]:
    """
    Generate one sentence per original sentence node using GPT-2 with beam search.

    node_to_choices should map node id -> list of (word, pos, score); the top entry is used.
    """
    if model is None or tokenizer is None:
        raise ValueError("tokenizer and model are required")

    sent_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "sentence"]
    # sort by numeric part if available
    def _sent_key(sid: str):
        try:
            return int(sid.split(":")[1])
        except Exception:
            return sid

    sent_nodes = sorted(sent_nodes, key=_sent_key)

    prompts: List[str] = []
    roles_per_sentence: List[List[Tuple[str, str, str]]] = []

    for sid in sent_nodes:
        roles = []
        for nbr in G.successors(sid):
            if nbr in node_to_choices and node_to_choices[nbr]:
                word, pos, _ = node_to_choices[nbr][0]
                kind = G.nodes[nbr].get("kind", "concept")
                roles.append((kind, word, pos))
        if not roles:
            sent_text = G.nodes[sid].get("text", "")
            if not sent_text:
                roles_per_sentence.append([])
                prompts.append("")
                continue
            roles = [("sentence", sent_text, "")]

        role_desc = "; ".join(f"{k} -> '{w}' ({p})" for k, w, p in roles)
        prompt = (
            "Write one short, clear sentence using these roles. "
            "Keep language simple. Roles: " + role_desc + "\nSentence:"
        )
        roles_per_sentence.append(roles)
        prompts.append(prompt)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: List[str] = [""] * len(prompts)

    if torch is not None:
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids,
                attention_mask=attn_mask,
                num_beams=beam_size,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for i, text in enumerate(decoded):
            prompt = prompts[i]
            if text.startswith(prompt):
                text = text[len(prompt) :]
            text = text.strip()
            if (not text) or ("->" in text and "Sentence" in prompt) or text.startswith("Roles:"):
                text = _simple_sentence_from_roles(roles_per_sentence[i])
            outputs[i] = text
    else:
        # Fallback to serial generation if torch is unavailable
        for i, prompt in enumerate(prompts):
            if not prompt:
                outputs[i] = ""
                continue
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            gen = model.generate(
                input_ids,
                num_beams=beam_size,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt) :]
            text = text.strip()
            if (not text) or ("->" in text and "Sentence" in prompt) or text.startswith("Roles:"):
                text = _simple_sentence_from_roles(roles_per_sentence[i])
            outputs[i] = text

    return outputs


def _mask_templates(words: List[str], mask_lengths: Sequence[int], max_candidates: int = 32) -> List[str]:
    """Build candidate templates with varying [MASK] spans between role words."""
    if len(words) <= 1:
        return [" ".join(words).strip()]
    gaps = len(words) - 1
    combos = list(itertools.product(mask_lengths, repeat=gaps))
    combos.sort(key=lambda c: (sum(c), c))
    templates: List[str] = []
    for combo in combos[:max_candidates]:
        parts: List[str] = []
        for idx, word in enumerate(words):
            parts.append(word)
            if idx < gaps:
                parts.extend(["[MASK]"] * combo[idx])
        templates.append(" ".join(parts))
    return templates


def _score_fill_mask_batch(templates: List[str], tokenizer, model, device: str = "cpu") -> List[Tuple[str, float]]:
    """Fill masks with argmax tokens and return decoded text with summed log-probability."""
    if torch is None:
        raise ImportError("torch is required for masked LM scoring")
    if not templates:
        return []
    enc = tokenizer(templates, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no mask token")
    mask_positions = input_ids == mask_id
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    log_probs = outputs.logits.log_softmax(dim=-1)

    decoded: List[Tuple[str, float]] = []
    for idx in range(input_ids.size(0)):
        mask_idx = mask_positions[idx].nonzero(as_tuple=False).flatten()
        token_ids = input_ids[idx].clone()
        if mask_idx.numel() == 0:
            decoded.append((tokenizer.decode(token_ids, skip_special_tokens=True).strip(), float("-inf")))
            continue
        score = 0.0
        for pos in mask_idx:
            lp_vec = log_probs[idx, pos]
            best_id = int(torch.argmax(lp_vec).item())
            score += float(lp_vec[best_id].item())
            token_ids[pos] = best_id
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        decoded.append((text, score))
    return decoded


def fill_mask_sentences_from_graph(
    G,
    node_to_choices: Dict[str, List[Tuple[str, str, float]]],
    tokenizer,
    model,
    mask_lengths: Sequence[int] = (1, 2, 3),
    max_candidates: int = 32,
    device: str = "cpu",
) -> List[str]:
    """
    Regenerate one sentence per sentence node using a masked-LM scorer.

    For each gap between role words, tries multiple consecutive [MASK] lengths,
    scores candidates in batch, and keeps the highest scoring fill per sentence.
    Falls back to deterministic clause if scoring fails.
    """
    if tokenizer is None or model is None:
        raise ValueError("tokenizer and model are required for BERT regen")

    sent_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "sentence"]

    def _sent_key(sid: str):
        try:
            return int(sid.split(":")[1])
        except Exception:
            return sid

    sent_nodes = sorted(sent_nodes, key=_sent_key)
    outputs: List[str] = []

    for sid in sent_nodes:
        roles = _roles_for_sentence_node(G, sid, node_to_choices)
        words = [w for _, w, _ in roles if w]
        if not words:
            outputs.append("")
            continue
        templates = _mask_templates(words, mask_lengths=mask_lengths, max_candidates=max_candidates)
        try:
            scored = _score_fill_mask_batch(templates, tokenizer, model, device=device)
        except Exception:
            outputs.append(_deterministic_clause(roles))
            continue
        if not scored:
            outputs.append(_deterministic_clause(roles))
            continue
        best_text, best_score = max(scored, key=lambda t: t[1])
        if best_score == float("-inf"):
            outputs.append(_deterministic_clause(roles))
        else:
            outputs.append(best_text)

    return outputs
