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
    args = parser.parse_args()

    result = ingest_movie(
        movie_path=args.movie,
        movie_id=args.movie_id,
        fps=args.fps,
        semantic_k=args.semantic_k,
        max_frames=args.max_frames,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
