from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from PIL import Image

from .query_decomposer import DecomposedQuery, QueryDecomposer

if TYPE_CHECKING:
    from ingestion.embedder import CLIPEmbedder
    from vector_store import ChromaEmbeddingStore


class SearchError(RuntimeError):
    """Raised when retrieval query construction or execution fails."""


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


@dataclass(slots=True)
class SearchInput:
    text: str | None = None
    image: Image.Image | None = None


@dataclass(slots=True)
class SearchContext:
    visual_query: str | None
    dialogue_query: str | None
    query_mode: str


class QuerySearcher:
    """Build multimodal CLIP query embeddings and execute Chroma top-k search."""

    def __init__(
        self,
        embedder: "CLIPEmbedder" | None = None,
        store: "ChromaEmbeddingStore" | None = None,
        decomposer: QueryDecomposer | None = None,
        chroma_dir: str | Path = "chroma_db",
    ) -> None:
        if embedder is None:
            from ingestion.embedder import CLIPEmbedder

            self.embedder = CLIPEmbedder()
        else:
            self.embedder = embedder
        if store is None:
            from vector_store import ChromaEmbeddingStore

            self.store = ChromaEmbeddingStore(persist_directory=chroma_dir)
        else:
            self.store = store
        self.decomposer = decomposer or QueryDecomposer()

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        if not text:
            return ""
        return " ".join(text.split()).strip()

    def _decompose_text(self, text: str) -> DecomposedQuery:
        return self.decomposer.decompose(text)

    def build_query_embedding(self, search_input: SearchInput) -> tuple[np.ndarray, SearchContext]:
        text = self._normalize_text(search_input.text)
        image = search_input.image

        if not text and image is None:
            raise ValueError("at least one of text or image must be provided")

        if text and image is not None:
            parts = self._decompose_text(text)
            text_emb = self.embedder.encode_text(parts.visual)
            image_emb = self.embedder.encode_image(image)
            merged = _l2_normalize(0.5 * text_emb + 0.5 * image_emb)
            context = SearchContext(
                visual_query=parts.visual,
                dialogue_query=parts.dialogue,
                query_mode="multimodal",
            )
            return merged, context

        if text:
            parts = self._decompose_text(text)
            embedding = self.embedder.encode_text(parts.visual)
            context = SearchContext(
                visual_query=parts.visual,
                dialogue_query=parts.dialogue,
                query_mode="text",
            )
            return embedding, context

        assert image is not None
        embedding = self.embedder.encode_image(image)
        context = SearchContext(
            visual_query=None,
            dialogue_query=None,
            query_mode="image",
        )
        return embedding, context

    def search(
        self,
        movie_id: str,
        search_input: SearchInput,
        top_k: int = 12,
    ) -> dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_embedding, context = self.build_query_embedding(search_input)
        raw_results = self.store.query_detailed(movie_id=movie_id, query_embedding=query_embedding, top_k=top_k)

        hits: list[dict[str, Any]] = []
        for item in raw_results:
            node_id = int(item["node_id"])
            score = float(item["score"])
            metadata = dict(item.get("metadata", {}))
            hits.append(
                {
                    "node_id": node_id,
                    "visual_score": score,
                    "metadata": metadata,
                }
            )

        return {
            "embedding": query_embedding,
            "context": context,
            "hits": hits,
        }


def search_movie(
    movie_id: str,
    text: str | None = None,
    image: Image.Image | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    searcher = QuerySearcher()
    return searcher.search(movie_id=movie_id, search_input=SearchInput(text=text, image=image), top_k=top_k)
