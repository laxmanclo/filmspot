from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    movie_path: str = Field(..., description="Absolute or relative path to movie file")
    movie_id: str | None = Field(default=None, description="Optional explicit movie id")
    fps: float = Field(default=1.0, gt=0)
    semantic_k: int = Field(default=5, gt=0)
    max_frames: int | None = Field(default=None, gt=0)
    caption_model: str = Field(default="gemini-2.5-flash")
    caption_device: str | None = Field(default=None)
    caption_batch_size: int = Field(default=4, gt=0)
    caption_max_new_tokens: int = Field(default=30, gt=0)


class IngestResponse(BaseModel):
    movie_id: str
    node_count: int
    duration_sec: float
    fps: float
    semantic_k: int
    caption_model: str
    index_dir: str
    movie_path: str
    indexed_at: str


class SceneResult(BaseModel):
    start_t: float
    end_t: float
    node_ids: list[int]
    caption: str
    transcript: str
    visual_score: float
    transcript_score: float
    final_score: float
    conflict: bool


class QueryResponse(BaseModel):
    context: dict[str, Any]
    scenes: list[SceneResult]
    synthesis: str | None = None


class MoviesResponse(BaseModel):
    movies: list[dict[str, Any]]


class IngestJobStartResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    movie_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class QueryHistoryEntry(BaseModel):
    id: str
    created_at: str
    query_text: str | None = None
    query_image: str | None = None
    scenes: list[SceneResult]


class QueryHistoryResponse(BaseModel):
    movie_id: str
    entries: list[QueryHistoryEntry]
