"""Ingestion utilities for FilmSpot."""

from .caption_generator import generate_captions
from .frame_extractor import extract_frames

__all__ = ["extract_frames", "generate_captions"]
