"""Retrieval engine modules for FilmSpot Phase 3."""

from .fusion import bm25_scores, fuse_candidates
from .graph_traversal import TraversalConfig, expand_hits
from .pipeline import RetrievalEngine
from .query_decomposer import DecomposedQuery, QueryDecomposer, decompose_query
from .reranker import rerank_scenes
from .searcher import QuerySearcher, SearchContext, SearchInput, search_movie

__all__ = [
    "DecomposedQuery",
    "QueryDecomposer",
    "decompose_query",
    "SearchInput",
    "SearchContext",
    "QuerySearcher",
    "search_movie",
    "RetrievalEngine",
    "TraversalConfig",
    "expand_hits",
    "bm25_scores",
    "fuse_candidates",
    "rerank_scenes",
]
