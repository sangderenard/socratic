
"""
Minimal demo script for the socratic_graph project. Legacy/kept for reference;
prefer using the library functions (socratic_graph.graph_with_embeddings, etc.).

Usage:
    python demo.py
"""

from socratic_graph import (
    load_models,
    text_to_graph_with_types,
    draw_graph,
)


def main():
    text = """
    In this conversation, truth means best concordance with observed data shape.
    I want you to prioritize internal consistency over comfort.
    Sometimes I want you to just make me feel better even if it breaks the rules.
    """
    print("Loading models...")
    nlp, sbert = load_models()

    print("Building graph...")
    G = text_to_graph_with_types(text, nlp, sbert)

    print("Nodes:")
    for n, data in G.nodes(data=True):
        print(n, "->", data)

    print("\nEdges:")
    for u, v, data in G.edges(data=True):
        print(f"{u} -> {v}:", data)

    print("\nDrawing graph window...")
    draw_graph(G)


if __name__ == "__main__":
    main()
