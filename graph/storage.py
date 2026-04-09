from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np

from .schema import SentenceNode


class GraphStorageError(RuntimeError):
    """Raised when graph save/load operations fail."""


def _sort_node_ids(node_ids: Iterable[Any]) -> list[Any]:
    try:
        return sorted(node_ids)
    except TypeError:
        return sorted(node_ids, key=str)


def _extract_node_embedding(attrs: dict[str, Any]) -> np.ndarray:
    if "image_emb" in attrs:
        emb = np.asarray(attrs["image_emb"], dtype=np.float32)
    elif isinstance(attrs.get("data"), SentenceNode):
        emb = np.asarray(attrs["data"].image_emb, dtype=np.float32)
    else:
        raise GraphStorageError("Node is missing image embedding data (image_emb or SentenceNode.data)")

    if emb.ndim != 1:
        raise GraphStorageError("Each node embedding must be a 1D vector")
    return emb


def save_graph(
    graph: nx.Graph,
    output_dir: str | Path,
    graph_filename: str = "graph.pkl",
    embeddings_filename: str = "frame_embeddings.npy",
    metadata_filename: str = "metadata.json",
    vector_ids: Sequence[str] | None = None,
) -> dict[str, Path]:
    """
    Save graph + aligned embeddings to disk.

    Storage layout (Phase 2 compatible):
      - graph.pkl               (graph structure + lightweight node attrs)
      - frame_embeddings.npy    (node-aligned embedding matrix)
      - metadata.json           (node order + optional node<->vector mapping)

    Embeddings are removed from the pickled graph for compact storage and restored via loader.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    node_order = _sort_node_ids(graph.nodes())
    embeddings: list[np.ndarray] = []

    for node_id in node_order:
        attrs = graph.nodes[node_id]
        embeddings.append(_extract_node_embedding(attrs))

    emb_matrix = np.stack(embeddings).astype(np.float32) if embeddings else np.zeros((0, 0), dtype=np.float32)

    if vector_ids is not None and len(vector_ids) != len(node_order):
        raise ValueError("vector_ids length must match number of graph nodes")

    graph_to_save = graph.copy()
    for node_id in node_order:
        node_attrs = graph_to_save.nodes[node_id]
        node_attrs.pop("image_emb", None)
        node_attrs.pop("data", None)

    graph_path = out_dir / graph_filename
    emb_path = out_dir / embeddings_filename
    meta_path = out_dir / metadata_filename

    with graph_path.open("wb") as f:
        pickle.dump(graph_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(emb_path, emb_matrix)

    metadata: dict[str, Any] = {
        "node_order": node_order,
        "embedding_dim": int(emb_matrix.shape[1]) if emb_matrix.size else 0,
        "node_count": len(node_order),
        "embeddings_file": embeddings_filename,
        "graph_file": graph_filename,
    }

    if vector_ids is not None:
        metadata["node_to_vector_id"] = {str(node_id): vector_ids[idx] for idx, node_id in enumerate(node_order)}

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {"graph": graph_path, "embeddings": emb_path, "metadata": meta_path}


def load_graph(
    input_dir: str | Path,
    graph_filename: str = "graph.pkl",
    embeddings_filename: str = "frame_embeddings.npy",
    metadata_filename: str = "metadata.json",
) -> tuple[nx.Graph, np.ndarray, dict[str, Any]]:
    """
    Load graph and reattach embeddings/SentenceNode objects.

    Returns:
      (graph, embeddings, metadata)
    """
    in_dir = Path(input_dir).expanduser().resolve()
    graph_path = in_dir / graph_filename
    emb_path = in_dir / embeddings_filename
    meta_path = in_dir / metadata_filename

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    with graph_path.open("rb") as f:
        graph: nx.Graph = pickle.load(f)

    embeddings = np.load(emb_path).astype(np.float32)
    if embeddings.ndim != 2:
        raise GraphStorageError("Loaded embeddings must have shape (N, D)")

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    node_order = metadata.get("node_order") if metadata else None
    if node_order is None:
        node_order = _sort_node_ids(graph.nodes())

    if len(node_order) != embeddings.shape[0]:
        raise GraphStorageError(
            f"Node count ({len(node_order)}) does not match embedding rows ({embeddings.shape[0]})"
        )

    node_to_vector = metadata.get("node_to_vector_id", {}) if metadata else {}

    for idx, node_id in enumerate(node_order):
        if node_id not in graph.nodes:
            raise GraphStorageError(f"Node id '{node_id}' from metadata not found in graph")

        emb = embeddings[idx]
        attrs = graph.nodes[node_id]
        attrs["image_emb"] = emb

        t = float(attrs.get("t", 0.0))
        caption = str(attrs.get("caption", ""))
        transcript = str(attrs.get("transcript", ""))
        attrs["data"] = SentenceNode(t=t, image_emb=emb, caption=caption, transcript=transcript)

        vec_id = node_to_vector.get(str(node_id))
        if vec_id is not None:
            attrs["vector_id"] = vec_id

    return graph, embeddings, metadata
