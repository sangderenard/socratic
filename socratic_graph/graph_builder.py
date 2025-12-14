
"""
Graph construction utilities.

This module glues together:
- spaCy for POS / dependency parsing
- sentence-transformers for semantic classification
- networkx for the actual graph structure
"""

from typing import Dict, List, Tuple
import networkx as nx

from .models import (
    classify_statement,
    classify_relation,
    CATEGORY_PROTOTYPES,
    REL_TEMPLATES,
    build_category_vectors,
    normalize_token_id,
)


def sentence_to_graph_components(sent, model, rel_vecs) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, str, Dict]]]:
    """
    Given a spaCy Span (sentence), return:
    - nodes: list of (node_id, attrs_dict)
    - edges: list of (src_id, dst_id, attrs_dict)

    Heuristics:
    - nouns/pronouns as concept nodes (normalized ids so pronouns are stable)
    - adjectives/adverbs as modifier nodes attached to their head concept
    - subject-verb-object edges as relations; fallback verb node when no object
    """
    nodes: Dict[str, Dict] = {}
    edges: List[Tuple[str, str, Dict]] = []

    def add_node(key: str, **attrs):
        if key not in nodes:
            nodes[key] = {"label": key, **attrs}
        else:
            nodes[key].update(attrs)

    # Concept nodes (normalized pronouns)
    for token in sent:
        if token.pos_ in ("NOUN", "PROPN", "PRON"):
            key = normalize_token_id(token)
            add_node(key, pos=token.pos_, kind="concept")

    # Modifier nodes (ADJ/ADV attached to concepts)
    for token in sent:
        if token.pos_ in ("ADJ", "ADV") and token.head.pos_ in ("NOUN", "PROPN", "PRON"):
            concept_key = normalize_token_id(token.head)
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
                            model,
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


def text_to_graph_with_types(text: str, nlp, sbert) -> nx.DiGraph:
    """
    Build a networkx DiGraph from input text.

    Each sentence becomes a "sentence" node with:
    - stmt_type: precept/value/fact/preference/self_state/unknown
    - stmt_conf: confidence
    and links to its main concept nodes via "about" edges.

    Concept and relation nodes are derived via sentence_to_graph_components.
    """
    doc = nlp(text)
    G = nx.DiGraph()

    # Precompute category vectors for statement and relation types
    cat_vecs = build_category_vectors(sbert, CATEGORY_PROTOTYPES)
    rel_vecs = build_category_vectors(sbert, REL_TEMPLATES)

    prev_sent_id = None

    for sent in doc.sents:
        sent_text = sent.text.strip()

        # classify sentence type
        stmt_type, conf = classify_statement(sent_text, sbert, cat_vecs)

        # extract sentence-level graph components
        nodes, edges = sentence_to_graph_components(sent, sbert, rel_vecs)

        # add nodes
        for nid, attrs in nodes:
            if nid in G.nodes:
                G.nodes[nid].update(attrs)
            else:
                G.add_node(nid, **attrs)

        # add edges
        for src, dst, attrs in edges:
            if G.has_edge(src, dst):
                existing = G[src][dst].get("rel", [])
                if not isinstance(existing, list):
                    existing = [existing]
                existing.append(attrs.get("rel", ""))
                G[src][dst]["rel"] = existing
            else:
                G.add_edge(src, dst, **attrs)

        # sentence-level node
        sent_id = f"sent:{sent.start}"
        G.add_node(
            sent_id,
            kind="sentence",
            stmt_type=stmt_type,
            stmt_conf=conf,
            text=sent_text,
        )

        # sequential edge between sentences (higher stiffness downstream)
        if prev_sent_id is not None:
            G.add_edge(prev_sent_id, sent_id, rel="next_sentence", k=2.0)
        prev_sent_id = sent_id

        # link sentence node to main concepts (include pronouns, normalized ids)
        main_concepts = [
            normalize_token_id(t) for t in sent if t.pos_ in ("NOUN", "PROPN", "PRON")
        ]
        for c in main_concepts:
            if c in G.nodes:
                G.add_edge(sent_id, c, rel="about")

    return G
