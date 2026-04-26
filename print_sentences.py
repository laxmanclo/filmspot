from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph.storage import load_graph

INDEX_DIR = Path("data/index/sample")

graph, _, metadata = load_graph(INDEX_DIR)

node_order = metadata.get("node_order", sorted(graph.nodes()))

print(f"{'#':<5} {'time(s)':<10} {'caption':<80} transcript")
print("-" * 130)

for node_id in node_order:
    attrs = graph.nodes[node_id]
    t = float(attrs.get("t", 0.0))
    caption = str(attrs.get("caption", "")).strip()
    transcript = str(attrs.get("transcript", "")).strip()
    print(f"{node_id:<5} {t:<10.1f} {caption:<80} {transcript}")
