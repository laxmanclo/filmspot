from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

APP_NAME = "filmspot"
HF_CACHE_DIR = "/root/.cache/huggingface"
DATA_DIR = "/workspace/data"
INDEX_DIR = "/workspace/data/index"
CHROMA_DIR = "/workspace/chroma_db"
REPO_DIR = "/root/filmspot"

app = modal.App(APP_NAME)

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path=REPO_DIR)
)

hf_cache_volume = modal.Volume.from_name("filmspot-hf-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name("filmspot-artifacts", create_if_missing=True)

_gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
_secrets = [modal.Secret.from_dict({"GEMINI_API_KEY": _gemini_key})] if _gemini_key else []


@app.function(
    image=runtime_image,
    volumes={HF_CACHE_DIR: hf_cache_volume, "/workspace": artifact_volume},
    secrets=_secrets,
    timeout=60 * 60,
    cpu=4,
)
def ingest_remote(
    movie_path: str,
    movie_id: str | None = None,
    fps: float = 1.0,
    semantic_k: int = 5,
    max_frames: int | None = None,
    caption_model: str = "gemini-2.5-flash",
    caption_device: str | None = None,
    caption_batch_size: int = 4,
    caption_max_new_tokens: int = 30,
) -> dict:
    artifact_volume.reload()
    os.chdir("/workspace")
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    resolved_movie_path = str(Path(movie_path))
    if not Path(resolved_movie_path).is_absolute():
        resolved_movie_path = str(Path("/workspace") / resolved_movie_path)

    from api.service import ingest_movie

    result = ingest_movie(
        movie_path=resolved_movie_path,
        movie_id=movie_id,
        fps=fps,
        semantic_k=semantic_k,
        max_frames=max_frames,
        caption_model=caption_model,
        caption_device=caption_device,
        caption_batch_size=caption_batch_size,
        caption_max_new_tokens=caption_max_new_tokens,
    )

    hf_cache_volume.commit()
    artifact_volume.commit()
    return result


@app.function(
    image=runtime_image,
    volumes={HF_CACHE_DIR: hf_cache_volume, "/workspace": artifact_volume},
    secrets=_secrets,
    timeout=20 * 60,
    cpu=2,
)
def query_remote(
    movie_id: str,
    text: str | None = None,
    image_path: str | None = None,
    top_k: int = 12,
    top_scenes: int = 3,
) -> dict:
    artifact_volume.reload()
    os.chdir("/workspace")
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    from PIL import Image

    from api.service import query_movie

    resolved_image_path: str | None = None
    if image_path:
        candidate = Path(image_path)
        if candidate.is_absolute():
            resolved_image_path = str(candidate)
        else:
            resolved_image_path = str(Path("/workspace") / candidate)

    image = Image.open(resolved_image_path).convert("RGB") if resolved_image_path else None
    result = query_movie(
        movie_id=movie_id,
        text=text,
        image=image,
        top_k=top_k,
        top_scenes=top_scenes,
    )

    return {"context": result["context"], "scenes": result["scenes"], "synthesis": result.get("synthesis")}


@app.local_entrypoint()
def main(
    action: str = "ingest",
    movie_path: str = "",
    movie_id: str = "sample",
    text: str = "",
    image_path: str = "",
    fps: float = 1.0,
    semantic_k: int = 5,
    max_frames: int = 0,
    top_k: int = 12,
    top_scenes: int = 3,
    caption_model: str = "gemini-2.5-flash",
    caption_device: str = "",
    caption_batch_size: int = 4,
    caption_max_new_tokens: int = 30,
) -> None:
    if action not in {"ingest", "query"}:
        raise ValueError("action must be either 'ingest' or 'query'")

    if action == "ingest":
        if not movie_path:
            raise ValueError("Provide --movie-path for ingest")

        path = Path(movie_path)
        if not path.exists():
            raise FileNotFoundError(f"Movie path not found locally: {movie_path}")

        remote_movie_path = f"data/movies/{path.name}"

        with artifact_volume.batch_upload(force=True) as batch:
            batch.put_file(path, remote_movie_path)

        result = ingest_remote.remote(
            movie_path=remote_movie_path,
            movie_id=movie_id or None,
            fps=fps,
            semantic_k=semantic_k,
            max_frames=max_frames if max_frames > 0 else None,
            caption_model=caption_model,
            caption_device=caption_device or None,
            caption_batch_size=caption_batch_size,
            caption_max_new_tokens=caption_max_new_tokens,
        )
        print(json.dumps(result, indent=2))
        return

    if not movie_id:
        raise ValueError("Provide --movie-id for query")
    if not text and not image_path:
        raise ValueError("Provide --text and/or --image-path for query")

    remote_image_path: str | None = None
    if image_path:
        image = Path(image_path)
        if not image.exists():
            raise FileNotFoundError(f"Image path not found locally: {image_path}")
        remote_image_path = f"data/queries/{image.name}"
        with artifact_volume.batch_upload(force=True) as batch:
            batch.put_file(image, remote_image_path)

    result = query_remote.remote(
        movie_id=movie_id,
        text=text or None,
        image_path=remote_image_path,
        top_k=top_k,
        top_scenes=top_scenes,
    )
    print(json.dumps(result, indent=2))
