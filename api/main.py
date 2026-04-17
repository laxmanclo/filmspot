from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .schemas import IngestRequest, IngestResponse, MoviesResponse, QueryResponse
from .service import ingest_movie, list_movies, query_movie


app = FastAPI(title="FilmSpot API", version="0.1.0")


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
        return QueryResponse(context=result["context"], scenes=result["scenes"], synthesis=result.get("synthesis"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/movies", response_model=MoviesResponse)
def movies() -> MoviesResponse:
    return MoviesResponse(movies=list_movies())
