from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path
from typing import Any

from PIL import Image

from graph import build_sentence_graph, build_sentence_nodes, load_graph, save_graph
from ingestion import extract_frames, generate_captions, transcribe_audio
from ingestion.embedder import encode_frames
from retrieval import RetrievalEngine
from vector_store import ChromaEmbeddingStore


_INDEX_ROOT = Path("data/index")
_CHROMA_DIR = Path("chroma_db")
_MOVIES_ROOT = Path("data/movies")
_HISTORY_ROOT = Path("data/query_history")
_INGEST_JOBS: dict[str, dict[str, Any]] = {}
_INGEST_LOCK = threading.Lock()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("movie_id must contain at least one alphanumeric character")
    return cleaned.lower()


def _movie_id_from_path(movie_path: Path) -> str:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    return _slugify(f"{movie_path.stem}_{timestamp}")


def _index_dir(movie_id: str) -> Path:
    return _INDEX_ROOT / _slugify(movie_id)


def _metadata_path(movie_id: str) -> Path:
    return _index_dir(movie_id) / "metadata.json"


def _history_path(movie_id: str) -> Path:
    return _HISTORY_ROOT / f"{_slugify(movie_id)}.json"


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _synthesize_scenes(scenes: list[dict[str, Any]]) -> str:
    if not scenes:
        return "No confident scene match found. Try a more specific visual phrase or increase indexed frames."

    lines: list[str] = []
    for idx, scene in enumerate(scenes, start=1):
        start_t = float(scene.get("start_t", 0.0))
        end_t = float(scene.get("end_t", 0.0))
        caption = str(scene.get("caption", "")).strip()
        transcript = str(scene.get("transcript", "")).strip()

        detail = caption or transcript or "No caption/transcript details were available for this window."
        if caption and transcript:
            detail = f"{caption} Dialogue: {transcript}"

        lines.append(f"Scene {idx} ({start_t:.1f}s-{end_t:.1f}s): {detail}")

    return " ".join(lines)


def _init_ingest_job(job_id: str, *, movie_id: str | None = None) -> None:
    with _INGEST_LOCK:
        _INGEST_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Job queued",
            "movie_id": movie_id,
            "result": None,
            "error": None,
            "updated_at": _utc_now(),
        }


def _update_ingest_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    movie_id: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with _INGEST_LOCK:
        if job_id not in _INGEST_JOBS:
            return
        payload = _INGEST_JOBS[job_id]
        if status is not None:
            payload["status"] = status
        if progress is not None:
            payload["progress"] = max(0.0, min(100.0, float(progress)))
        if message is not None:
            payload["message"] = message
        if movie_id is not None:
            payload["movie_id"] = movie_id
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error
        payload["updated_at"] = _utc_now()


def _run_ingest_job(
    *,
    job_id: str,
    movie_path: str,
    movie_id: str,
    fps: float,
    semantic_k: int,
    max_frames: int | None,
    caption_model: str,
    caption_device: str | None,
    caption_batch_size: int,
    caption_max_new_tokens: int,
) -> None:
    try:
        _update_ingest_job(job_id, status="running", progress=5.0, message="Starting ingest", movie_id=movie_id)
        _update_ingest_job(job_id, progress=20.0, message="Extracting frames and transcribing audio")
        result = ingest_movie(
            movie_path=movie_path,
            movie_id=movie_id,
            fps=fps,
            semantic_k=semantic_k,
            max_frames=max_frames,
            caption_model=caption_model,
            caption_device=caption_device,
            caption_batch_size=caption_batch_size,
            caption_max_new_tokens=caption_max_new_tokens,
        )
        _update_ingest_job(
            job_id,
            status="completed",
            progress=100.0,
            message="Ingest completed",
            movie_id=result.get("movie_id", movie_id),
            result=result,
        )
    except Exception as exc:
        _update_ingest_job(
            job_id,
            status="failed",
            progress=100.0,
            message="Ingest failed",
            movie_id=movie_id,
            error=str(exc),
        )


def ingest_movie(
    movie_path: str | Path,
    movie_id: str | None = None,
    fps: float = 1.0,
    semantic_k: int = 5,
    max_frames: int | None = None,
    caption_model: str = "gemini-2.5-flash",
    caption_device: str | None = None,
    caption_batch_size: int = 4,
    caption_max_new_tokens: int = 30,
) -> dict[str, Any]:
    source = Path(movie_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Movie file not found: {source}")

    resolved_movie_id = _slugify(movie_id) if movie_id else _movie_id_from_path(source)
    out_dir = _index_dir(resolved_movie_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(source, fps=fps, max_frames=max_frames)
    if not frames:
        raise ValueError("No frames extracted; cannot ingest movie")

    captions = generate_captions(
        frames,
        model_name=caption_model,
        device=caption_device,
        batch_size=caption_batch_size,
        max_new_tokens=caption_max_new_tokens,
    )
    captions_map = {float(ts): text for ts, text in captions}
    transcripts_by_second = transcribe_audio(source)

    timestamps = [float(ts) for ts, _ in frames]
    frame_captions = [captions_map.get(float(ts), "") for ts in timestamps]

    embeddings = encode_frames(frames)

    nodes = build_sentence_nodes(
        timestamps=timestamps,
        image_embeddings=embeddings,
        captions=frame_captions,
        transcripts_by_second=transcripts_by_second,
    )
    graph = build_sentence_graph(nodes, semantic_k=semantic_k)

    store = ChromaEmbeddingStore(persist_directory=_CHROMA_DIR)
    vector_ids = store.upsert_embeddings(
        movie_id=resolved_movie_id,
        embeddings=embeddings,
        timestamps=timestamps,
        captions=frame_captions,
        transcripts=[transcripts_by_second.get(int(t), "") for t in timestamps],
        node_ids=list(range(len(timestamps))),
    )

    save_graph(graph=graph, output_dir=out_dir, vector_ids=vector_ids)

    duration_sec = float(max(timestamps)) if timestamps else 0.0
    metadata = {
        "movie_id": resolved_movie_id,
        "movie_path": str(source),
        "indexed_at": _utc_now(),
        "node_count": len(timestamps),
        "duration_sec": duration_sec,
        "fps": fps,
        "semantic_k": semantic_k,
        "caption_model": caption_model,
        "index_dir": str(out_dir),
    }

    with _metadata_path(resolved_movie_id).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def save_uploaded_movie(file_bytes: bytes, *, original_filename: str, movie_name: str | None = None) -> tuple[str, str, str]:
    _MOVIES_ROOT.mkdir(parents=True, exist_ok=True)

    source_name = movie_name or Path(original_filename).stem or "clip"
    slug = _slugify(source_name)
    suffix = Path(original_filename).suffix.lower() or ".mp4"
    target = _MOVIES_ROOT / f"{slug}{suffix}"

    if target.exists():
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
        target = _MOVIES_ROOT / f"{slug}_{stamp}{suffix}"

    target.write_bytes(file_bytes)
    inferred_movie_id = _slugify(target.stem)
    return str(target), inferred_movie_id, target.name


def start_ingest_job(
    *,
    movie_path: str,
    movie_id: str,
    fps: float,
    semantic_k: int,
    max_frames: int | None,
    caption_model: str,
    caption_device: str | None,
    caption_batch_size: int,
    caption_max_new_tokens: int,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    _init_ingest_job(job_id, movie_id=movie_id)

    worker = threading.Thread(
        target=_run_ingest_job,
        kwargs={
            "job_id": job_id,
            "movie_path": movie_path,
            "movie_id": movie_id,
            "fps": fps,
            "semantic_k": semantic_k,
            "max_frames": max_frames,
            "caption_model": caption_model,
            "caption_device": caption_device,
            "caption_batch_size": caption_batch_size,
            "caption_max_new_tokens": caption_max_new_tokens,
        },
        daemon=True,
    )
    worker.start()
    return get_ingest_job(job_id)


def get_ingest_job(job_id: str) -> dict[str, Any]:
    with _INGEST_LOCK:
        payload = _INGEST_JOBS.get(job_id)
        if payload is None:
            raise FileNotFoundError(f"Ingest job not found: {job_id}")
        return dict(payload)


def list_movies() -> list[dict[str, Any]]:
    _INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    movies: list[dict[str, Any]] = []
    for child in sorted(_INDEX_ROOT.iterdir()):
        if not child.is_dir():
            continue
        metadata_file = child / "metadata.json"
        if metadata_file.exists():
            with metadata_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["history_count"] = len(get_query_history(payload.get("movie_id", child.name), limit=10000))
            movies.append(payload)
        else:
            movies.append({"movie_id": child.name, "index_dir": str(child)})
    return movies


def get_movie_metadata(movie_id: str) -> dict[str, Any]:
    path = _metadata_path(_slugify(movie_id))
    if not path.exists():
        raise FileNotFoundError(f"Movie metadata not found for id: {movie_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_movie_stream_path(movie_id: str) -> tuple[Path, str]:
    metadata = get_movie_metadata(movie_id)
    movie_path = Path(metadata["movie_path"]).expanduser().resolve()
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie file does not exist on disk: {movie_path}")
    media_type, _ = guess_type(str(movie_path))
    return movie_path, media_type or "video/mp4"


def get_query_history(movie_id: str, limit: int = 50) -> list[dict[str, Any]]:
    history_file = _history_path(movie_id)
    if not history_file.exists():
        return []
    with history_file.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        return []
    if limit <= 0:
        return []
    return entries[-limit:]


def append_query_history(
    *,
    movie_id: str,
    query_text: str | None,
    query_image: str | None,
    scenes: list[dict[str, Any]],
) -> dict[str, Any]:
    _HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    history_file = _history_path(movie_id)
    existing = get_query_history(movie_id, limit=100000)

    entry = {
        "id": uuid.uuid4().hex,
        "created_at": _utc_now(),
        "query_text": query_text,
        "query_image": query_image,
        "scenes": scenes,
    }
    existing.append(entry)

    with history_file.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    return entry


def query_movie(
    movie_id: str,
    text: str | None = None,
    image: Image.Image | None = None,
    top_k: int = 12,
    top_scenes: int = 3,
) -> dict[str, Any]:
    resolved_movie_id = _slugify(movie_id)
    directory = _index_dir(resolved_movie_id)
    if not directory.exists():
        raise FileNotFoundError(f"Movie index not found: {directory}")

    graph, _, _ = load_graph(directory)
    engine = RetrievalEngine(chroma_dir=_CHROMA_DIR)
    result = engine.retrieve(
        movie_id=resolved_movie_id,
        graph=graph,
        text=text,
        image=image,
        top_k=top_k,
        top_scenes=top_scenes,
    )
    result["synthesis"] = _synthesize_scenes(result.get("scenes", []))
    return result
