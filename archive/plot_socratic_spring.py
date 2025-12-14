from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


def load_layout(path: str):
    """Load nodes/edges and positions from the JSON exported by socratic_spring_export.py."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    coords = np.array([n["pos"] for n in nodes], dtype=float)
    kinds = [n.get("kind", "concept") for n in nodes]
    ids = [n["id"] for n in nodes]

    return nodes, edges, coords, kinds, ids


def plot_3d(
    nodes,
    edges,
    coords,
    kinds,
    ids,
    show_edges: bool = True,
    max_labels: int = 0,
):
    """Simple 3D scatter plot of nodes + optional edges."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Separate sentence vs others
    coords = np.asarray(coords)
    kinds = np.asarray(kinds)

    is_sentence = kinds == "sentence"
    is_other = ~is_sentence

    # Colors
    # (Feel free to tweak colors to your own palette.)
    ax.scatter(
        coords[is_other, 0],
        coords[is_other, 1],
        coords[is_other, 2],
        s=10,
        alpha=0.5,
        label="other",
    )
    ax.scatter(
        coords[is_sentence, 0],
        coords[is_sentence, 1],
        coords[is_sentence, 2],
        s=20,
        alpha=0.9,
        marker="o",
        label="sentences",
    )

    # Optional edges
    if show_edges and edges:
        for e in edges:
            src = e["source"]
            tgt = e["target"]
            # Find indices
            try:
                i = ids.index(src)
                j = ids.index(tgt)
            except ValueError:
                continue
            xs = [coords[i, 0], coords[j, 0]]
            ys = [coords[i, 1], coords[j, 1]]
            zs = [coords[i, 2], coords[j, 2]]
            ax.plot(xs, ys, zs, linewidth=0.3, alpha=0.2)

    # Optional labels (for a subset of sentence nodes)
    if max_labels > 0:
        sent_indices = np.where(is_sentence)[0]
        # pick evenly spaced indices to avoid labeling everything
        if len(sent_indices) > 0:
            stride = max(1, len(sent_indices) // max_labels)
            for idx in sent_indices[::stride]:
                nid = ids[idx]
                text = nodes[idx].get("text", nid)
                # keep labels short
                if len(text) > 40:
                    text = text[:37] + "..."
                ax.text(
                    coords[idx, 0],
                    coords[idx, 1],
                    coords[idx, 2],
                    text,
                    fontsize=6,
                    alpha=0.7,
                )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    # Auto-scale to a roughly cubic view
    max_range = (coords.max(axis=0) - coords.min(axis=0)).max()
    mid = coords.mean(axis=0)
    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(m - max_range / 2.0, m + max_range / 2.0)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Quick 3D matplotlib visualization of socratic_spring_export JSON."
    )
    parser.add_argument(
        "json_path",
        help="Path to JSON file produced by socratic_spring_export.py",
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Do not render edges (only nodes).",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=0,
        help="Maximum number of sentence labels to draw (0 = no labels).",
    )

    args = parser.parse_args()

    nodes, edges, coords, kinds, ids = load_layout(args.json_path)
    plot_3d(
        nodes,
        edges,
        coords,
        kinds,
        ids,
        show_edges=not args.no_edges,
        max_labels=args.max_labels,
    )


if __name__ == "__main__":
    main()
