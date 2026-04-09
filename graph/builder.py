from __future__ import annotations

from typing import Mapping, Sequence

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .schema import SentenceNode


def build_sentence_nodes(
    timestamps: Sequence[float],
    image_embeddings: np.ndarray,
    captions: Sequence[str] | None = None,
    transcripts_by_second: Mapping[int, str] | None = None,
) -> list[SentenceNode]:
    """
    Create one `SentenceNode` per second from aligned ingestion outputs.

    Args:
        timestamps: Frame timestamps (seconds), one per node.
        image_embeddings: CLIP frame embeddings, shape (N, D).
        captions: Optional caption for each timestamp.
        transcripts_by_second: Optional map `{int_sec: transcript}` from Whisper.

    Returns:
        List[SentenceNode] with length N.
    """
    emb = np.asarray(image_embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("image_embeddings must have shape (N, D)")

    n = emb.shape[0]
    if len(timestamps) != n:
        raise ValueError("timestamps length must match number of embeddings")

    captions = captions or [""] * n
    if len(captions) != n:
        raise ValueError("captions length must match number of embeddings")

    transcripts_by_second = transcripts_by_second or {}

    nodes: list[SentenceNode] = []
    for i in range(n):
        t = float(timestamps[i])
        transcript = str(transcripts_by_second.get(int(t), ""))
        node = SentenceNode(
            t=t,
            image_emb=emb[i],
            caption=str(captions[i]),
            transcript=transcript,
        )
        nodes.append(node)

    return nodes


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def build_sentence_graph(nodes: Sequence[SentenceNode], semantic_k: int = 5) -> nx.Graph:
    """
    Build temporal sentence graph with:
      1) sequential edges i <-> i+1 (type='temporal', weight=1.0)
      2) semantic KNN edges (type='semantic', weight=cosine_sim), top-k per node,
         excluding self and adjacent nodes (|i-j| <= 1)
    """
    if semantic_k <= 0:
        raise ValueError("semantic_k must be > 0")

    ordered_nodes = sorted(nodes, key=lambda n: n.t)
    n = len(ordered_nodes)

    graph = nx.Graph()

    if n == 0:
        return graph

    embeddings = np.stack([node.image_emb for node in ordered_nodes]).astype(np.float32)

    for idx, node in enumerate(ordered_nodes):
        graph.add_node(
            idx,
            t=node.t,
            image_emb=node.image_emb,
            caption=node.caption,
            transcript=node.transcript,
            data=node,
        )

    # Sequential temporal edges
    for idx in range(n - 1):
        graph.add_edge(idx, idx + 1, type="temporal", weight=1.0)

    # Semantic KNN edges using sklearn NearestNeighbors as required.
    n_neighbors = min(n, semantic_k + 3)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    for i in range(n):
        added = 0
        for dist, j in zip(distances[i], indices[i]):
            j = int(j)
            if j == i:
                continue
            if abs(i - j) <= 1:  # exclude adjacent +/-1
                continue

            sim = _cosine_similarity(embeddings[i], embeddings[j])

            if graph.has_edge(i, j):
                edge_type = graph.edges[i, j].get("type")
                if edge_type == "semantic":
                    graph.edges[i, j]["weight"] = max(float(graph.edges[i, j]["weight"]), sim)
            else:
                graph.add_edge(i, j, type="semantic", weight=sim)

            added += 1
            if added >= semantic_k:
                break

    return graph
