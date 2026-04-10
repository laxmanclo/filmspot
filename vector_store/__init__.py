"""Vector store utilities for FilmSpot."""

__all__ = ["ChromaEmbeddingStore"]


def __getattr__(name: str):
	if name == "ChromaEmbeddingStore":
		from .chroma_store import ChromaEmbeddingStore

		return ChromaEmbeddingStore
	raise AttributeError(f"module 'vector_store' has no attribute {name!r}")
