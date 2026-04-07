from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image


class FrameExtractionError(RuntimeError):
    """Raised when frame extraction fails."""


def _require_ffmpeg_binaries() -> None:
    """Ensure ffmpeg and ffprobe are available on PATH."""
    missing = [binary for binary in ("ffmpeg", "ffprobe") if shutil.which(binary) is None]
    if missing:
        missing_display = ", ".join(missing)
        raise FrameExtractionError(
            f"Missing required binary/binaries: {missing_display}. "
            "Install ffmpeg (which includes ffprobe) and ensure both are on PATH."
        )


def _probe_video(video_path: Path) -> Tuple[int, int, float | None]:
    """Return (width, height, duration_sec) for the first video stream."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise FrameExtractionError(f"ffprobe failed: {exc.stderr.strip() or exc}") from exc

    try:
        payload = json.loads(proc.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        duration_raw = payload.get("format", {}).get("duration")
        duration = float(duration_raw) if duration_raw is not None else None
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrameExtractionError("Unable to parse ffprobe output for video dimensions.") from exc

    return width, height, duration


def _validated_time_window(start_sec: float, end_sec: float | None, duration: float | None) -> Tuple[float, float | None]:
    if start_sec < 0:
        raise ValueError("start_sec must be >= 0")

    effective_end = end_sec
    if duration is not None:
        effective_end = min(duration, end_sec) if end_sec is not None else duration

    if effective_end is not None and effective_end <= start_sec:
        return start_sec, start_sec

    return start_sec, effective_end


def _build_ffmpeg_cmd(
    video_path: Path,
    fps: float,
    source_width: int,
    source_height: int,
    start_sec: float,
    end_sec: float | None,
    resize: Tuple[int, int] | None,
) -> Tuple[List[str], int, int]:
    if fps <= 0:
        raise ValueError("fps must be > 0")

    out_width, out_height = (resize if resize is not None else (source_width, source_height))
    if out_width <= 0 or out_height <= 0:
        raise ValueError("resize dimensions must be > 0")

    filters = [f"fps={fps}"]
    if resize is not None:
        filters.append(f"scale={out_width}:{out_height}")

    cmd: List[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.6f}"])

    cmd.extend(["-i", str(video_path)])

    if end_sec is not None:
        cmd.extend(["-to", f"{end_sec:.6f}"])

    cmd.extend(
        [
            "-vf",
            ",".join(filters),
            "-vsync",
            "vfr",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )

    return cmd, out_width, out_height


def extract_frames(
    video_path: str | Path,
    fps: float = 1.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    resize: Tuple[int, int] | None = None,
    max_frames: int | None = None,
) -> List[Tuple[float, Image.Image]]:
    """
    Extract frames from a video using ffmpeg and return a list of (timestamp_sec, PIL.Image).

    Args:
        video_path: Input movie path.
        fps: Sampling rate, default 1 frame per second.
        start_sec: Start offset in seconds.
        end_sec: Optional inclusive end timestamp in seconds.
        resize: Optional (width, height) output resolution.
        max_frames: Optional hard cap on number of returned frames.

    Returns:
        List of tuples: (timestamp_sec, PIL.Image.Image)
    """
    _require_ffmpeg_binaries()

    path = Path(video_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    source_width, source_height, duration = _probe_video(path)
    start_sec, end_sec = _validated_time_window(start_sec, end_sec, duration)

    if end_sec is not None and math.isclose(start_sec, end_sec):
        return []

    cmd, out_width, out_height = _build_ffmpeg_cmd(
        video_path=path,
        fps=fps,
        source_width=source_width,
        source_height=source_height,
        start_sec=start_sec,
        end_sec=end_sec,
        resize=resize,
    )

    frame_size = out_width * out_height * 3
    if frame_size <= 0:
        raise FrameExtractionError("Invalid output frame size computed.")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise FrameExtractionError(f"Failed to start ffmpeg: {exc}") from exc

    frames: List[Tuple[float, Image.Image]] = []
    idx = 0
    stopped_early = False

    assert proc.stdout is not None
    while True:
        raw = proc.stdout.read(frame_size)
        if not raw:
            break
        if len(raw) != frame_size:
            proc.kill()
            raise FrameExtractionError("Received incomplete frame bytes from ffmpeg.")

        image = Image.frombytes("RGB", (out_width, out_height), raw)
        ts = start_sec + (idx / fps)
        frames.append((round(ts, 3), image))
        idx += 1

        if max_frames is not None and idx >= max_frames:
            stopped_early = True
            proc.terminate()
            break

    _, stderr = proc.communicate()
    if not stopped_early and proc.returncode not in (0, None):
        raise FrameExtractionError(stderr.decode("utf-8", errors="replace").strip() or "ffmpeg failed")

    return frames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video file using ffmpeg.")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling rate (frames per second)")
    parser.add_argument("--start", type=float, default=0.0, help="Start timestamp in seconds")
    parser.add_argument("--end", type=float, default=None, help="End timestamp in seconds")
    parser.add_argument("--width", type=int, default=None, help="Optional output width")
    parser.add_argument("--height", type=int, default=None, help="Optional output height")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum frames to extract")
    args = parser.parse_args()

    resize_dims = (args.width, args.height) if args.width and args.height else None
    output = extract_frames(
        video_path=args.video,
        fps=args.fps,
        start_sec=args.start,
        end_sec=args.end,
        resize=resize_dims,
        max_frames=args.max_frames,
    )

    print(f"Extracted {len(output)} frame(s)")
    for ts, img in output:
        print(f"t={ts:.3f}s, size={img.size}")
