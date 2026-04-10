"""Ingestion utilities for FilmSpot."""

__all__ = ["extract_frames", "generate_captions", "transcribe_audio", "encode_frames", "encode_text", "encode_image"]


def __getattr__(name: str):
	if name == "extract_frames":
		from .frame_extractor import extract_frames

		return extract_frames
	if name == "generate_captions":
		from .caption_generator import generate_captions

		return generate_captions
	if name == "transcribe_audio":
		from .audio_transcriber import transcribe_audio

		return transcribe_audio
	if name == "encode_frames":
		from .embedder import encode_frames

		return encode_frames
	if name == "encode_text":
		from .embedder import encode_text

		return encode_text
	if name == "encode_image":
		from .embedder import encode_image

		return encode_image
	raise AttributeError(f"module 'ingestion' has no attribute {name!r}")
