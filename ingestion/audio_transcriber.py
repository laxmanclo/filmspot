from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import torch
import whisper


class AudioTranscriptionError(RuntimeError):
    """Raised when Whisper transcription fails."""


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_token(text: str) -> str:
    return " ".join(text.strip().split())


def bucket_word_segments_per_second(words: Iterable[Mapping[str, Any]]) -> Dict[int, str]:
    """
    Bucket Whisper word-level segments into `int(second) -> joined text`.

    Each word is placed into `floor(word.start)`.
    """
    buckets: dict[int, list[str]] = defaultdict(list)

    for word in words:
        start = word.get("start")
        raw_text = str(word.get("word", ""))
        token = _normalize_token(raw_text)

        if start is None or not token:
            continue

        try:
            sec = int(math.floor(float(start)))
        except (TypeError, ValueError):
            continue

        buckets[sec].append(token)

    return {sec: " ".join(tokens) for sec, tokens in sorted(buckets.items()) if tokens}


def align_transcription_to_second_buckets(result: Mapping[str, Any]) -> Dict[int, str]:
    """Align Whisper transcription output into per-second transcript buckets."""
    words: list[Mapping[str, Any]] = []

    for seg in result.get("segments", []) or []:
        seg_words = seg.get("words")
        if isinstance(seg_words, list):
            words.extend(seg_words)

    if words:
        return bucket_word_segments_per_second(words)

    # Fallback path for models/runs without word-level timestamps.
    buckets: dict[int, list[str]] = defaultdict(list)
    for seg in result.get("segments", []) or []:
        start = seg.get("start")
        text = _normalize_token(str(seg.get("text", "")))
        if start is None or not text:
            continue
        try:
            sec = int(math.floor(float(start)))
        except (TypeError, ValueError):
            continue
        buckets[sec].append(text)

    return {sec: " ".join(tokens) for sec, tokens in sorted(buckets.items()) if tokens}


@lru_cache(maxsize=4)
def _cached_model(model_name: str, device: str) -> Any:
    return whisper.load_model(model_name, device=device)


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = "small",
    language: str | None = None,
    device: str | None = None,
    task: str = "transcribe",
) -> Dict[int, str]:
    """
    Transcribe audio with Whisper and return `dict[int_sec -> transcript_str]`.

    Uses `word_timestamps=True` and aligns words into per-second buckets.
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    chosen_device = device or _default_device()

    try:
        model = _cached_model(model_name=model_name, device=chosen_device)
        result = model.transcribe(
            str(path),
            task=task,
            language=language,
            word_timestamps=True,
            verbose=False,
            fp16=chosen_device.startswith("cuda"),
        )
    except Exception as exc:  # pragma: no cover - API boundary consistency
        raise AudioTranscriptionError(
            f"Whisper transcription failed for '{path.name}' with model '{model_name}' on '{chosen_device}': {exc}"
        ) from exc

    return align_transcription_to_second_buckets(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe an audio/video file with OpenAI Whisper")
    parser.add_argument("audio", type=str, help="Path to audio/video file")
    parser.add_argument("--model", type=str, default="small", help="Whisper model (tiny/base/small/medium/large)")
    parser.add_argument("--language", type=str, default=None, help="Optional language code, e.g., en")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g., cpu or cuda")
    args = parser.parse_args()

    out = transcribe_audio(
        audio_path=args.audio,
        model_name=args.model,
        language=args.language,
        device=args.device,
    )

    print(f"Buckets: {len(out)}")
    for sec, text in out.items():
        print(f"{sec}: {text}")
