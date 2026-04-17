from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from api.service import query_movie


def main() -> None:
    parser = argparse.ArgumentParser(description="Query an indexed movie in FilmSpot")
    parser.add_argument("--movie", required=True, type=str, help="Movie ID from ingest step")
    parser.add_argument("--text", type=str, default=None, help="Optional text query")
    parser.add_argument("--image", type=str, default=None, help="Optional reference image path")
    parser.add_argument("--top-k", type=int, default=12, help="Top vector hits before traversal")
    parser.add_argument("--top-scenes", type=int, default=3, help="Number of final merged scenes")
    args = parser.parse_args()

    if not args.text and not args.image:
        raise SystemExit("Provide at least one of --text or --image")

    image = Image.open(args.image).convert("RGB") if args.image else None

    result = query_movie(
        movie_id=args.movie,
        text=args.text,
        image=image,
        top_k=args.top_k,
        top_scenes=args.top_scenes,
    )

    print(json.dumps({"context": result["context"], "scenes": result["scenes"], "synthesis": result.get("synthesis")}, indent=2))


if __name__ == "__main__":
    main()
