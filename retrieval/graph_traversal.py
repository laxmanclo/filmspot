from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx


@dataclass(slots=True)
class TraversalConfig:
    depth: int = 2
    semantic_decay: float = 0.7
    temporal_decay: float = 0.9


def _edge_decay(edge_type: str, config: TraversalConfig) -> float:
    if edge_type == "semantic":
        return config.semantic_decay
    return config.temporal_decay


def _node_payload(graph: nx.Graph, node_id: int) -> dict[str, Any]:
    attrs = graph.nodes[node_id]
    return {
        "t": float(attrs.get("t", 0.0)),
        "caption": str(attrs.get("caption", "")),
        "transcript": str(attrs.get("transcript", "")),
    }


def expand_hits(
    graph: nx.Graph,
    seed_hits: Iterable[dict[str, Any]],
    config: TraversalConfig | None = None,
) -> list[dict[str, Any]]:
    """
    BFS expansion from seed nodes with edge-type decay.

    - semantic edges: score *= 0.7
    - temporal edges: score *= 0.9
    - depth default: 2
    """
    cfg = config or TraversalConfig()
    if cfg.depth < 0:
        raise ValueError("depth must be >= 0")

    best_score: dict[int, float] = {}
    best_source: dict[int, int] = {}

    for hit in seed_hits:
        node_id = int(hit["node_id"])
        base_score = float(hit.get("visual_score", 0.0))
        if node_id not in graph:
            continue

        queue: deque[tuple[int, int, float]] = deque([(node_id, 0, base_score)])
        local_best: dict[tuple[int, int], float] = {(node_id, 0): base_score}

        while queue:
            current, depth, score = queue.popleft()

            if score > best_score.get(current, float("-inf")):
                best_score[current] = score
                best_source[current] = node_id

            if depth >= cfg.depth:
                continue

            for neighbor in graph.neighbors(current):
                edge = graph.edges[current, neighbor]
                edge_type = str(edge.get("type", "temporal"))
                decayed = score * _edge_decay(edge_type=edge_type, config=cfg)

                state = (neighbor, depth + 1)
                if decayed <= local_best.get(state, float("-inf")):
                    continue

                local_best[state] = decayed
                queue.append((neighbor, depth + 1, decayed))

    rows: list[dict[str, Any]] = []
    for node_id, score in best_score.items():
        payload = _node_payload(graph, node_id)
        rows.append(
            {
                "node_id": int(node_id),
                "visual_score": float(score),
                "source_node_id": int(best_source.get(node_id, node_id)),
                **payload,
            }
        )

    rows.sort(key=lambda item: item["visual_score"], reverse=True)
    return rows
