from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.service import ingest_movie


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a movie into FilmSpot index")
    parser.add_argument("movie", type=str, help="Path to movie file")
    parser.add_argument("--movie-id", type=str, default=None, help="Optional explicit movie ID")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate")
    parser.add_argument("--semantic-k", type=int, default=5, help="Semantic KNN edges per node")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for fast testing")
    parser.add_argument(
        "--caption-model",
        type=str,
        default="gemini-2.5-flash",
        help="Caption model id (Gemini default; legacy BLIP also supported)",
    )
    parser.add_argument("--caption-device", type=str, default=None, help="Caption model device, e.g. cpu/cuda")
    parser.add_argument("--caption-batch-size", type=int, default=4, help="Caption generation batch size")
    parser.add_argument("--caption-max-new-tokens", type=int, default=30, help="Caption generation max tokens")
    args = parser.parse_args()

    result = ingest_movie(
        movie_path=args.movie,
        movie_id=args.movie_id,
        fps=args.fps,
        semantic_k=args.semantic_k,
        max_frames=args.max_frames,
        caption_model=args.caption_model,
        caption_device=args.caption_device,
        caption_batch_size=args.caption_batch_size,
        caption_max_new_tokens=args.caption_max_new_tokens,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
