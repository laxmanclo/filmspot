# FilmSpot

Multimodal temporal video retrieval system that returns timestamped scenes for text/image queries.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure `ffmpeg` + `ffprobe` are installed and available on `PATH`.

For Gemini decomposition, set an API key:

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
