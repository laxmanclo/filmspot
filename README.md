# FilmSpot

Multimodal temporal video retrieval system that returns timestamped scenes for text/image queries.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure `ffmpeg` + `ffprobe` are installed and available on `PATH`.

For Gemini decomposition and Gemini-based frame captioning, set an API key:

```bash
export GEMINI_API_KEY="your_key_here"
```

## CLI

Ingest a movie:

```bash
python3 scripts/ingest_movie.py data/movies/sample.mp4 --movie-id sample
```

Query by text:

```bash
python3 scripts/query.py --movie sample --text "when does the chase begin"
```

Query by image:

```bash
python3 scripts/query.py --movie sample --image reference_frame.jpg
```

## Run on Modal (faster setup for heavy models)

Use Modal to run ingest/query remotely and persist model cache + index artifacts.

Install tooling:

```bash
uv pip install -r requirements.txt
uv pip install modal
modal setup
```

Ingest from local video to remote index (uses Gemini captioning by default):

```bash
modal run modal_app.py --action ingest --movie-path data/movies/sample.mp4 --movie-id sample --max-frames 120
```

Run text retrieval on the same indexed movie:

```bash
modal run modal_app.py --action query --movie-id sample --text "when does the chase begin"
```

Optional: switch back to BLIP captioning if you want local HuggingFace caption models:

```bash
modal run modal_app.py --action ingest --movie-path data/movies/sample.mp4 --movie-id sample --caption-model Salesforce/blip-image-captioning-base
```

## API

Run FastAPI server:

```bash
uvicorn api.main:app --reload
```

Endpoints:
- `POST /ingest`
- `POST /query`
- `GET /movies`

## Demo

```bash
python3 demo/app.py
```

This opens a Gradio UI for ingest + query.
