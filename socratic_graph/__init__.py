
"""
socratic_graph package

Tools to:
- Parse text with spaCy
- Classify sentences and relations with sentence-transformers
- Build a crude concept/precept graph with networkx
- Visualize the resulting network
"""

from .models import (
    build_category_vectors,
    classify_sentence_semantic,
    classify_statement,
    classify_relation,
    load_models,
    normalize_token_id,
)

from .graph_builder import (
    sentence_to_graph_components,
    text_to_graph_with_types,
)

from .visualize import draw_graph

from .data import (
    graph_with_embeddings,
    compute_node_embeddings,
    edge_distance_table,
)

from .generative import (
    suggest_simple_words_for_nodes,
    load_wordfreq_lexicon,
    load_gpt2_small,
    generate_sentences_from_graph,
    deterministic_sentences_from_graph,
    load_bert_fill_mask,
    fill_mask_sentences_from_graph,
)
