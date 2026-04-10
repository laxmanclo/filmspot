from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(slots=True)
class _Cluster:
    node_ids: list[int]
    start_t: float
    end_t: float
    best_hit: dict[str, Any]


def _to_time_sorted(hits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(hits, key=lambda item: (float(item.get("t", 0.0)), -float(item.get("final_score", 0.0))))


def _cluster_hits(hits: Sequence[dict[str, Any]], merge_window_sec: float) -> list[_Cluster]:
    if not hits:
        return []

    ordered = _to_time_sorted(hits)
    clusters: list[_Cluster] = []

    for hit in ordered:
        t = float(hit.get("t", 0.0))
        node_id = int(hit["node_id"])

        if not clusters:
            clusters.append(_Cluster(node_ids=[node_id], start_t=t, end_t=t, best_hit=hit))
            continue

        current = clusters[-1]
        if t - current.end_t <= merge_window_sec:
            current.node_ids.append(node_id)
            current.end_t = max(current.end_t, t)
            if float(hit.get("final_score", 0.0)) > float(current.best_hit.get("final_score", 0.0)):
                current.best_hit = hit
            continue

        clusters.append(_Cluster(node_ids=[node_id], start_t=t, end_t=t, best_hit=hit))

    return clusters


def rerank_scenes(
    fused_hits: Sequence[dict[str, Any]],
    merge_window_sec: float = 5.0,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """
    Merge nearby per-second hits into scene windows and return top-N scenes.

    Merge policy: consecutive candidate timestamps with gap <= merge_window_sec.
    Ranking policy: best final score of each merged cluster.
    """
    if merge_window_sec < 0:
        raise ValueError("merge_window_sec must be >= 0")
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    clusters = _cluster_hits(hits=fused_hits, merge_window_sec=merge_window_sec)

    scenes: list[dict[str, Any]] = []
    for cluster in clusters:
        best = cluster.best_hit
        scenes.append(
            {
                "start_t": float(cluster.start_t),
                "end_t": float(cluster.end_t),
                "node_ids": list(cluster.node_ids),
                "caption": str(best.get("caption", "")),
                "transcript": str(best.get("transcript", "")),
                "visual_score": float(best.get("visual_score", 0.0)),
                "transcript_score": float(best.get("transcript_score", 0.0)),
                "final_score": float(best.get("final_score", 0.0)),
                "conflict": bool(best.get("conflict", False)),
            }
        )

    scenes.sort(key=lambda item: item["final_score"], reverse=True)
    return scenes[:top_n]
