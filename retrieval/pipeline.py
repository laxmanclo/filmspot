from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx
from PIL import Image

from .fusion import fuse_candidates
from .graph_traversal import TraversalConfig, expand_hits
from .reranker import rerank_scenes
from .searcher import QuerySearcher, SearchInput


class RetrievalEngine:
    """End-to-end Phase 3 retrieval pipeline over an indexed movie graph."""

    def __init__(self, chroma_dir: str | Path = "chroma_db") -> None:
        self.searcher = QuerySearcher(chroma_dir=chroma_dir)

    def retrieve(
        self,
        movie_id: str,
        graph: nx.Graph,
        text: str | None = None,
        image: Image.Image | None = None,
        top_k: int = 12,
        traversal: TraversalConfig | None = None,
        merge_window_sec: float = 2.0,
        top_scenes: int = 3,
    ) -> dict[str, Any]:
        if top_scenes <= 0:
            raise ValueError("top_scenes must be > 0")

        search = self.searcher.search(
            movie_id=movie_id,
            search_input=SearchInput(text=text, image=image),
            top_k=top_k,
        )

        expanded = expand_hits(graph=graph, seed_hits=search["hits"], config=traversal)
        context = search["context"]

        fused = fuse_candidates(
            graph=graph,
            candidates=expanded,
            dialogue_query=context.dialogue_query,
        )

        scenes = rerank_scenes(
            fused_hits=fused,
            merge_window_sec=merge_window_sec,
            top_n=top_scenes,
        )

        return {
            "context": asdict(context),
            "seed_hits": search["hits"],
            "expanded_hits": expanded,
            "fused_hits": fused,
            "scenes": scenes,
        }
