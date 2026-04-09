from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import chromadb
import numpy as np


class ChromaStoreError(RuntimeError):
    """Raised when ChromaDB operations fail."""


def _collection_name(movie_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in movie_id.strip())
    if not cleaned:
        raise ValueError("movie_id must not be empty")
    return f"movie_{cleaned}"


class ChromaEmbeddingStore:
    """Persistent ChromaDB wrapper for frame embeddings."""

    def __init__(self, persist_directory: str | Path = "chroma_db") -> None:
        self.persist_directory = Path(persist_directory).expanduser().resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    def _get_or_create_collection(self, movie_id: str):
        return self.client.get_or_create_collection(
            name=_collection_name(movie_id),
            metadata={"hnsw:space": "cosine", "movie_id": movie_id},
        )

    def upsert_embeddings(
        self,
        movie_id: str,
        embeddings: np.ndarray,
        timestamps: Sequence[float],
        captions: Sequence[str] | None = None,
        transcripts: Sequence[str] | None = None,
        node_ids: Sequence[int] | None = None,
    ) -> list[str]:
        """Upsert frame embeddings and metadata for a movie and return vector IDs."""
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, D)")

        n = embeddings.shape[0]
        if len(timestamps) != n:
            raise ValueError("timestamps length must match number of embeddings")

        captions = captions or [""] * n
        transcripts = transcripts or [""] * n
        node_ids = node_ids or list(range(n))

        if len(captions) != n or len(transcripts) != n or len(node_ids) != n:
            raise ValueError("captions/transcripts/node_ids lengths must match number of embeddings")

        ids = [f"{movie_id}:{int(node_ids[i])}" for i in range(n)]
        metadatas = [
            {
                "movie_id": movie_id,
                "node_id": int(node_ids[i]),
                "t": float(timestamps[i]),
                "caption": str(captions[i]),
                "transcript": str(transcripts[i]),
            }
            for i in range(n)
        ]

        collection = self._get_or_create_collection(movie_id)
        collection.upsert(
            ids=ids,
            embeddings=embeddings.astype(np.float32).tolist(),
            metadatas=metadatas,
        )
        return ids

    def query(self, movie_id: str, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Query a movie collection and return top-k matches with cosine-like scores."""
        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must have shape (D,)")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        collection = self._get_or_create_collection(movie_id)
        resp = collection.query(
            query_embeddings=[query_embedding.astype(np.float32).tolist()],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        ids = (resp.get("ids") or [[]])[0]
        dists = (resp.get("distances") or [[]])[0]
        metas = (resp.get("metadatas") or [[]])[0]

        results: list[dict[str, Any]] = []
        for vec_id, dist, meta in zip(ids, dists, metas):
            distance = float(dist)
            score = 1.0 - distance
            item_meta = meta or {}
            results.append(
                {
                    "id": vec_id,
                    "node_id": int(item_meta.get("node_id", -1)),
                    "t": float(item_meta.get("t", 0.0)),
                    "caption": str(item_meta.get("caption", "")),
                    "transcript": str(item_meta.get("transcript", "")),
                    "distance": distance,
                    "score": score,
                }
            )
        return results

    def restore_embeddings(self, movie_id: str) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
        """Restore all embeddings for a movie from ChromaDB."""
        collection = self._get_or_create_collection(movie_id)
        data = collection.get(include=["embeddings", "metadatas"])
        raw_ids = data.get("ids")
        raw_embeddings = data.get("embeddings")
        raw_metadatas = data.get("metadatas")

        ids = list(raw_ids) if raw_ids is not None else []
        metadatas = list(raw_metadatas) if raw_metadatas is not None else []

        if raw_embeddings is None:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            embeddings = np.array(raw_embeddings, dtype=np.float32)

        if embeddings.ndim == 1 and embeddings.size == 0:
            embeddings = np.zeros((0, 0), dtype=np.float32)

        return list(ids), embeddings, list(metadatas)

    def count(self, movie_id: str) -> int:
        collection = self._get_or_create_collection(movie_id)
        return int(collection.count())

    def delete_movie(self, movie_id: str) -> None:
        self.client.delete_collection(name=_collection_name(movie_id))
