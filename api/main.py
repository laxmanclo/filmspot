from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .schemas import (
    IngestJobStartResponse,
    IngestJobStatusResponse,
    IngestRequest,
    IngestResponse,
    MoviesResponse,
    QueryHistoryResponse,
    QueryResponse,
)
from .service import (
    append_query_history,
    get_ingest_job,
    get_movie_stream_path,
    get_query_history,
    ingest_movie,
    list_movies,
    query_movie,
    save_uploaded_movie,
    start_ingest_job,
)


app = FastAPI(title="FilmSpot API", version="0.1.0")
_ROOT = Path(__file__).resolve().parents[1]
_WEB_DIR = _ROOT / "demo" / "web"
_WEB_ASSETS_DIR = _WEB_DIR / "assets"

if _WEB_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_WEB_ASSETS_DIR), name="assets")


@app.get("/", include_in_schema=False)
def web_home() -> FileResponse:
    index_file = _WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI is not available")
    return FileResponse(index_file)


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    try:
        result = ingest_movie(
            movie_path=payload.movie_path,
            movie_id=payload.movie_id,
            fps=payload.fps,
            semantic_k=payload.semantic_k,
            max_frames=payload.max_frames,
            caption_model=payload.caption_model,
            caption_device=payload.caption_device,
            caption_batch_size=payload.caption_batch_size,
            caption_max_new_tokens=payload.caption_max_new_tokens,
        )
        return IngestResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/upload", response_model=IngestJobStartResponse)
async def ingest_upload(
    movie_name: str = Form(...),
    clip: UploadFile = File(...),
    fps: float = Form(default=1.0),
    semantic_k: int = Form(default=5),
    max_frames: int | None = Form(default=None),
    caption_model: str = Form(default="gemini-2.5-flash"),
    caption_device: str | None = Form(default=None),
    caption_batch_size: int = Form(default=4),
    caption_max_new_tokens: int = Form(default=30),
) -> IngestJobStartResponse:
    if not clip.filename:
        raise HTTPException(status_code=400, detail="Uploaded clip must have a filename")
    if not movie_name.strip():
        raise HTTPException(status_code=400, detail="movie_name is required")

    try:
        file_bytes = await clip.read()
        saved_path, suggested_movie_id, _ = save_uploaded_movie(
            file_bytes,
            original_filename=clip.filename,
            movie_name=movie_name.strip(),
        )

        job = start_ingest_job(
            movie_path=saved_path,
            movie_id=suggested_movie_id,
            fps=fps,
            semantic_k=semantic_k,
            max_frames=max_frames,
            caption_model=caption_model,
            caption_device=caption_device,
            caption_batch_size=caption_batch_size,
            caption_max_new_tokens=caption_max_new_tokens,
        )
        return IngestJobStartResponse(
            job_id=job["job_id"],
            status=job["status"],
            progress=float(job["progress"]),
            message=str(job["message"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/ingest/jobs/{job_id}", response_model=IngestJobStatusResponse)
def ingest_job_status(job_id: str) -> IngestJobStatusResponse:
    try:
        payload = get_ingest_job(job_id)
        return IngestJobStatusResponse(**payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
async def query(
    movie_id: str = Form(...),
    text: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
    top_k: int = Form(default=12),
    top_scenes: int = Form(default=3),
) -> QueryResponse:
    if not text and image is None:
        raise HTTPException(status_code=400, detail="Provide text and/or image")

    query_image: Image.Image | None = None
    if image is not None:
        try:
            raw = await image.read()
            query_image = Image.open(BytesIO(raw)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="Uploaded image is not a valid image file") from exc

    try:
        result = query_movie(
            movie_id=movie_id,
            text=text,
            image=query_image,
            top_k=top_k,
            top_scenes=top_scenes,
        )
        append_query_history(
            movie_id=movie_id,
            query_text=text,
            query_image=image.filename if image else None,
            scenes=result["scenes"],
        )
        return QueryResponse(context=result["context"], scenes=result["scenes"], synthesis=result.get("synthesis"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/movies", response_model=MoviesResponse)
def movies() -> MoviesResponse:
    return MoviesResponse(movies=list_movies())


@app.get("/movies/{movie_id}/stream", include_in_schema=False)
def movie_stream(movie_id: str) -> FileResponse:
    try:
        stream_path, media_type = get_movie_stream_path(movie_id)
        return FileResponse(stream_path, media_type=media_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/movies/{movie_id}/history", response_model=QueryHistoryResponse)
def movie_query_history(movie_id: str, limit: int = 50) -> QueryHistoryResponse:
    try:
        entries = get_query_history(movie_id=movie_id, limit=limit)
        return QueryHistoryResponse(movie_id=movie_id, entries=entries)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
