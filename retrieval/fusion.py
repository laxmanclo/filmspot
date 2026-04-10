from __future__ import annotations

from typing import Any, Sequence

import networkx as nx


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _concat_text(caption: str, transcript: str) -> str:
    return f"{caption} {transcript}".strip()


def bm25_scores(dialogue_query: str, docs: Sequence[str]) -> list[float]:
    try:
        from rank_bm25 import BM25Okapi
    except Exception as exc:
        raise RuntimeError("rank-bm25 is required for transcript scoring. Install dependencies from pyproject.toml.") from exc

    query = " ".join(dialogue_query.split()).strip()
    if not docs:
        return []
    if not query:
        return [0.0] * len(docs)

    tokenized_docs = [_tokenize(doc) for doc in docs]
    tokenized_query = _tokenize(query)

    if not tokenized_query:
        return [0.0] * len(docs)

    bm25 = BM25Okapi(tokenized_docs)
    raw = [float(score) for score in bm25.get_scores(tokenized_query)]

    max_val = max(raw) if raw else 0.0
    if max_val <= 1e-12:
        return [0.0] * len(raw)
    return [float(score / max_val) for score in raw]


def fuse_candidates(
    graph: nx.Graph,
    candidates: Sequence[dict[str, Any]],
    dialogue_query: str | None,
    visual_weight: float = 0.65,
    transcript_weight: float = 0.35,
    conflict_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Fuse visual retrieval with lexical transcript/caption BM25.

    final = visual_weight * visual + transcript_weight * transcript
    conflict = abs(visual - transcript) > conflict_threshold
    """
    if not 0.0 <= visual_weight <= 1.0:
        raise ValueError("visual_weight must be in [0, 1]")
    if not 0.0 <= transcript_weight <= 1.0:
        raise ValueError("transcript_weight must be in [0, 1]")
    if abs((visual_weight + transcript_weight) - 1.0) > 1e-6:
        raise ValueError("visual_weight + transcript_weight must equal 1.0")

    docs: list[str] = []
    for item in candidates:
        node_id = int(item["node_id"])
        attrs = graph.nodes[node_id]
        docs.append(_concat_text(str(attrs.get("caption", "")), str(attrs.get("transcript", ""))))

    transcript_scores = bm25_scores(dialogue_query or "", docs)

    fused: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates):
        node_id = int(item["node_id"])
        attrs = graph.nodes[node_id]
        visual_score = float(item.get("visual_score", 0.0))
        transcript_score = float(transcript_scores[idx]) if idx < len(transcript_scores) else 0.0

        final_score = visual_weight * visual_score + transcript_weight * transcript_score
        conflict = abs(visual_score - transcript_score) > conflict_threshold

        fused.append(
            {
                "node_id": node_id,
                "t": float(attrs.get("t", 0.0)),
                "caption": str(attrs.get("caption", "")),
                "transcript": str(attrs.get("transcript", "")),
                "visual_score": visual_score,
                "transcript_score": transcript_score,
                "final_score": float(final_score),
                "conflict": bool(conflict),
            }
        )

    fused.sort(key=lambda row: row["final_score"], reverse=True)
    return fused
