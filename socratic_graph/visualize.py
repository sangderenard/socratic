
"""
Visualization helpers for the socratic_graph network.
"""

from typing import Tuple
import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(G: nx.DiGraph, figsize: Tuple[int, int] = (12, 9)) -> None:
    """
    Draw a networkx DiGraph using a spring layout.

    - Node color encodes kind (concept/modifier/sentence/other)
    - Sentence nodes with stmt_type="precept" are highlighted
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

    plt.axis("off")
    plt.tight_layout()
    plt.show()
