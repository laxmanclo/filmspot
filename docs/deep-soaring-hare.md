# FilmSpot — Multimodal Temporal Video Retrieval System

## Context
Assignment project building a system that answers natural language queries ("when does the villain reveal himself?") with precise movie timestamps. Core idea from professor: use CLIP + BLIP. We're layering a **temporal sentence graph** on top, plus Whisper audio, LLM query decomposition, and multimodal score fusion as the "crazy stuff" to score full marks.

Repo at `/home/laxman/Desktop/filmspot` is currently empty (just a LICENSE).

---

## Why CLIP and BLIP — Correct Role of Each

**CLIP's actual job:** CLIP has a shared image-text embedding space. A frame image of a car chase and the text "car chase" land close together. This means:
- At ingest: run CLIP's **image encoder** on every frame → store image embeddings as graph node vectors
- At query (text): run CLIP's **text encoder** → cosine sim against frame image embeddings
- At query (image): run CLIP's **image encoder** → cosine sim against frame image embeddings
- Both query types work against the same stored frame embeddings — this is CLIP's whole point

**BLIP's actual job:** BLIP is a VQA/captioning model. It reads a frame and outputs a human-readable sentence like "a man in a suit runs through a crowded market". This caption:
- Goes into graph node metadata (shown in results as evidence)
- Is used by BM25 for text-based transcript matching alongside Whisper
- Gives the LLM decomposer richer context about what's visually in a scene

BLIP does NOT produce embeddings used for search. CLIP does.

---

## Architecture Overview

```
INGEST
Video File
    │
    ├─ ffmpeg ──────────────────── frames @ 1fps (PIL.Image each)
    │                                   │                    │
    │                               CLIP image encoder   BLIP-2 captioner
    │                                   │                    │
    │                           image embedding (512d)   caption string
    │                                   │                    │
    ├─ Whisper ─────────────────── transcript per second     │
    │                                   │                    │
    └───────────────────────────── Sentence Graph Node ──────┘
                                   {t, image_emb, caption, transcript}
                                   sequential edges (t↔t+1)
                                   KNN edges (top-5 by image_emb cosine sim)
                                   + persist vectors/metadata in ChromaDB

QUERY
    User input: text ("when does X happen")
             OR image (a screenshot/reference frame)
             OR both (image + text description together)
                    │
            ┌───────┴────────┐
            │                │
      text branch        image branch
      LLM decomposer     (skip decomposer,
      → visual_q           use image directly)
      → dialogue_q
            │                │
      CLIP text encoder  CLIP image encoder
            │                │
            └───────┬────────┘
                    │
             query embedding (512d)  ← same space as stored frame embeddings
                    │
       ChromaDB vector search against persisted frame embeddings → top-k hits
                    │
          graph traversal (expand via KNN + sequential edges)
                    │
          BM25 on dialogue_q vs (captions + transcripts) → transcript score
                    │
          fusion: 0.65 × visual_score + 0.35 × transcript_score
                    │
          confidence resolver: flag if |visual - transcript| > 0.3
                    │
          reranker: merge nodes within 5s → top-3 [start_t, end_t] scenes
                    │
          Return: timestamp range + matching caption + transcript + flag
```

---

## File Structure

```
filmspot/
├── ingestion/
│   ├── __init__.py
│   ├── frame_extractor.py      # ffmpeg → list[(t, PIL.Image)] @ 1fps
│   ├── caption_generator.py    # BLIP-2 → list[(t, caption_str)]
│   ├── audio_transcriber.py    # Whisper → dict[int_sec → transcript_str]
│   └── embedder.py             # CLIP image encoder → np.ndarray (N, 512)
│                               # also exposes encode_text() and encode_image() for queries
│
├── graph/
│   ├── __init__.py
│   ├── schema.py               # @dataclass SentenceNode {t, image_emb, caption, transcript}
│   ├── builder.py              # networkx graph: sequential edges + KNN edges on image_emb
│   └── storage.py              # save/load graph (pickle) + node/index maps for vector IDs

├── vector_store/
│   ├── __init__.py
│   └── chroma_store.py         # persistent ChromaDB client, upsert/query frame embeddings
│
├── retrieval/
│   ├── __init__.py
│   ├── query_decomposer.py     # Gemini API: text query → {visual_q, dialogue_q}
│   ├── searcher.py             # CLIP encode query (text or image) → cosine sim search
│   ├── graph_traversal.py      # expand hits via KNN + sequential edges with score decay
│   ├── fusion.py               # visual_score + BM25 transcript_score → final + conflict flag
│   └── reranker.py             # merge nearby hits → top-3 scene windows [start_t, end_t]
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI: POST /ingest, POST /query (text or image or both)
│   └── schemas.py              # Pydantic models
│
├── demo/
│   └── app.py                  # Gradio UI: upload movie → type query or upload ref image
│
├── scripts/
│   ├── ingest_movie.py         # CLI: python scripts/ingest_movie.py movie.mp4
│   └── query.py                # CLI: --text "..." or --image ref.jpg or both
│
├── data/
│   ├── movies/
│   └── index/                  # per movie: graph.pkl + metadata.json

├── chroma_db/                  # persistent ChromaDB directory (local)
│
├── requirements.txt
└── README.md
```

---

## Implementation Plan (Phase by Phase)

### Phase 1 — Ingestion Pipeline

- `frame_extractor.py`: `ffmpeg` subprocess → dump frames at 1fps → `list[(timestamp_sec, PIL.Image)]`
- `caption_generator.py`: load `Salesforce/blip2-opt-2.7b`, batch frames → `list[(t, caption_str)]`; captions go to graph metadata only, NOT used as search vectors
- `audio_transcriber.py`: `openai-whisper` on audio track, align word-level segments into per-second buckets → `dict[int_sec → str]`
- `embedder.py`:
  - `encode_frames(frames)` → run CLIP **image encoder** on each PIL.Image → `np.ndarray (N, 512)` — these are the stored search vectors
  - `encode_text(text: str)` → CLIP **text encoder** → `np.ndarray (512,)` — used at query time for text queries
  - `encode_image(img: PIL.Image)` → CLIP **image encoder** → `np.ndarray (512,)` — used at query time for image queries

### Phase 2 — Sentence Graph

- `schema.py`: `@dataclass SentenceNode { t: float, image_emb: np.ndarray, caption: str, transcript: str }`
- `builder.py`:
  - one node per second
  - **sequential edges** `t ↔ t+1`, `type="temporal"`, `weight=1.0`
  - **KNN edges**: `sklearn.NearestNeighbors` on all `image_emb` vectors → top-5 neighbors per node (excluding adjacent ±1s), `type="semantic"`, `weight=cosine_sim`
- `storage.py`: pickle graph (without embeddings in nodes to save space) + separate `frame_embeddings.npy` aligned by node index; loader reattaches
- `storage.py`: pickle graph (without heavy embeddings in nodes where possible) + persist node↔vector ID mapping metadata for Chroma lookup
- `vector_store/chroma_store.py`:
  - initialize `chromadb.PersistentClient(path="chroma_db")`
  - per-movie collection (e.g., `movie_<movie_id>`)
  - upsert `ids`, `embeddings`, and metadata `{t, caption, transcript, node_id}`
  - query top-k by embedding and return `(node_id, visual_score, metadata)`

### Phase 3 — Retrieval Engine

- `query_decomposer.py`: Gemini API prompt → given text query, return `{"visual": "person in shadows removing mask", "dialogue": "I was the killer all along"}`. If image query, skip this step — image IS the visual query.
- `searcher.py`:
  - accepts either `text: str` OR `image: PIL.Image` or both
  - if text: `embed = encode_text(visual_q)`
  - if image: `embed = encode_image(query_image)`
  - if both: `embed = 0.5 * encode_text(visual_q) + 0.5 * encode_image(query_image)` (averaged, renormalized)
  - ChromaDB similarity query of `embed` against persisted movie collection → top-k `(node_id, visual_score)`
- `graph_traversal.py`: BFS from each hit node up to depth=2; semantic edges decay score by `0.7×`, temporal edges by `0.9×`; accumulate best score per node
- `fusion.py`:
  - visual score: from searcher (CLIP cosine sim)
  - transcript score: BM25 on `dialogue_q` vs per-node `(caption + transcript)` text
  - final = `0.65 × visual + 0.35 × transcript`
  - conflict flag: `abs(visual - transcript) > 0.3` → attach `conflict: true` + both raw scores to result
- `reranker.py`: sort by final score, merge nodes within 5s window → return top-3 `{start_t, end_t, caption, transcript, visual_score, transcript_score, conflict}`

### Phase 4 — API

- `POST /ingest` → `{movie_path}` → runs pipeline → `{movie_id, node_count, duration_sec}`
- `POST /query` → multipart form: `movie_id` (str) + `text` (optional str) + `image` (optional file) → `{scenes: [...]}`
- `GET /movies` → list of indexed movie IDs

### Phase 5 — UI

- Build a good looking UI using NextJS
- The home page should list all the movie clips that are already ingested
- The home page should have an "Add Clip" button that allows the user to upload a new movie clip, give it a name and ingest it in the backend. Once the user enters the information and clicks "Add" it should show a progress bar of the movie being ingested
- Once the movie is ingested, it should appear in the home page
- When the user clicks on the ingested movie, it should bring up a movie player that is able to play the selected movie clip, like a standard video player with volume controls, seeker etc
- The control bar should have a chat button in the bottom right side. When the user clicks it, a chat panel should open up in the right hand side of the screen
- The user should be able to type in a query and and send it.
- Once the query is sent, the back end should process it and return all timestamps where the query matches
- The returned timestamps durations should be visible in the seek bar of the video, the user should be able to select one of the marked timestamp to jump to that point
- The history of queries asked for a given movie clip and the response timestamp should be stored and be visible across sessions 

---

## Key Dependencies (`requirements.txt`)

```
torch
torchvision
transformers          # BLIP-2
openai-clip           # CLIP (openai/clip-vit-base-patch32)
openai-whisper        # audio transcription
Pillow
ffmpeg-python
networkx
scikit-learn          # NearestNeighbors for KNN
numpy
chromadb             # persistent vector DB for frame embeddings
fastapi
uvicorn
gradio
google-generativeai   # Gemini API for query decomposition
pydantic
rank-bm25             # BM25 for transcript matching
tqdm
```

---

## The "Crazy" Demo Moment (for the prof)

**Cross-modal conflict resolution + image-as-query:**
1. User uploads a screenshot from a different movie with a similar scene → system finds the matching moment in the indexed film. Zero text needed.
2. When visual and audio disagree, system surfaces: `"Visual: 0.87 (dark room, figure in shadows) | Dialogue: 0.18 (scene is silent) | ⚠ conflict — visual prioritized"` — system is *reasoning about uncertainty*, not just retrieving.

---

## Verification

1. `pip install -r requirements.txt`
2. `python scripts/ingest_movie.py data/movies/sample.mp4` → produces `data/index/<id>/graph.pkl` + metadata and upserts embeddings into `chroma_db/`
3. `python scripts/query.py --movie <id> --text "when does the chase begin"` → returns top-3 timestamps
4. `python scripts/query.py --movie <id> --image reference_frame.jpg` → image-based retrieval
5. `uvicorn api.main:app --reload` → POST /ingest then POST /query via curl
6. `python demo/app.py` → Gradio UI in browser

---

## What we're NOT building
- No VideoMAE / TimeSformer
- No distributed/cloud vector infrastructure (single-node local ChromaDB is enough)
- No user auth
