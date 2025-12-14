
# socratic_precept_graph

A tiny experimental project to:

- Parse conversational text into a **crude concept graph** using spaCy.
- Classify sentences (precept/value/fact/preference/...) with **sentence-transformers**.
- Classify subject-verb-object relationships (causal/definition/preference/...) via embeddings.
- Build and visualize a **networkx** graph that you can later use for Socratic consequence checks.

This is intentionally lossy and clumsy: it's a starting point for exploring
"precept Socratic graphs" rather than a polished NLP library.

## Layout

- `socratic_graph/__init__.py` – convenience imports.
- `socratic_graph/models.py` – model loading and semantic classification helpers.
- `socratic_graph/graph_builder.py` – POS/dependency-based graph construction.
- `socratic_graph/visualize.py` – simple networkx/matplotlib visualization.
- `demo.py` / `newdemo.py` – legacy demos kept for reference; prefer calling the
  library API directly.
- `socratic_spring_export.py` – export graph data (pre-physics embeddings + edge
  distances) and optional mass–spring layout JSON.
- `gl_animator.py` – OpenGL + pygame spring/repulsion animator using a
  second-order (Velocity-Verlet) integrator; consumes the pre-physics JSON.

## Installation

Create and activate a virtualenv, then install dependencies:

```bash
pip install spacy sentence-transformers networkx matplotlib
python -m spacy download en_core_web_sm

# for the OpenGL animator
pip install pygame PyOpenGL
```

If you want a different spaCy model or SBERT backbone, adjust the calls
in `demo.py` or in your own scripts.

## Running the demo

```bash
# build graph + embeddings + edge distances (pre-physics)
python socratic_spring_export.py --pre-out prephysics.json --skip-physics

# or keep physics/export layout as before
python socratic_spring_export.py --out layout.json

# run the OpenGL animator on the pre-physics data
python gl_animator.py prephysics.json
```

`prephysics.json` contains:

- `nodes`: graph nodes with kind/text/lemma and SentenceTransformer embedding
  (resting position) per node.
- `edges`: graph edges with library-inferred relation labels and Euclidean
  distance between connected node embeddings.

Use these data to drive your own layout/visualization pipelines; physics and
matplotlib are optional and isolated.

