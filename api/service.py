from __future__ import annotations

import json
import re
from datetime import datetime, timezone
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


def ingest_movie(
    movie_path: str | Path,
    movie_id: str | None = None,
    fps: float = 1.0,
    semantic_k: int = 5,
    max_frames: int | None = None,
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

    captions = generate_captions(frames)
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
        "indexed_at": datetime.now(tz=timezone.utc).isoformat(),
        "node_count": len(timestamps),
        "duration_sec": duration_sec,
        "fps": fps,
        "semantic_k": semantic_k,
        "index_dir": str(out_dir),
    }

    with _metadata_path(resolved_movie_id).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


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
            movies.append(payload)
        else:
            movies.append({"movie_id": child.name, "index_dir": str(child)})
    return movies


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
    return result
