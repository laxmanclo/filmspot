from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image


class FrameExtractionError(RuntimeError):
    """Raised when frame extraction fails."""


def _resolve_ffmpeg_binaries() -> tuple[str, str | None]:
    """Resolve ffmpeg/ffprobe binaries from PATH or imageio-ffmpeg fallback."""
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")

    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:  # pragma: no cover - environment fallback
            raise FrameExtractionError(
                "Missing ffmpeg binary. Install system ffmpeg or Python package imageio-ffmpeg."
            ) from exc

    return ffmpeg_bin, ffprobe_bin


def _probe_video(video_path: Path, ffmpeg_bin: str, ffprobe_bin: str | None) -> Tuple[int, int, float | None]:
    """Return (width, height, duration_sec) for the first video stream."""
    if ffprobe_bin is not None:
        cmd = [
            ffprobe_bin,
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
            payload = json.loads(proc.stdout)
            stream = payload["streams"][0]
            width = int(stream["width"])
            height = int(stream["height"])
            duration_raw = payload.get("format", {}).get("duration")
            duration = float(duration_raw) if duration_raw is not None else None
            return width, height, duration
        except (subprocess.CalledProcessError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            pass

    fallback_cmd = [ffmpeg_bin, "-hide_banner", "-i", str(video_path)]
    proc = subprocess.run(fallback_cmd, capture_output=True, text=True)
    stderr = proc.stderr or ""

    size_match = re.search(r"(\d{2,5})x(\d{2,5})", stderr)
    if not size_match:
        raise FrameExtractionError("Unable to infer video dimensions from ffmpeg output.")

    width = int(size_match.group(1))
    height = int(size_match.group(2))

    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if duration_match:
        hours = float(duration_match.group(1))
        minutes = float(duration_match.group(2))
        seconds = float(duration_match.group(3))
        duration = hours * 3600 + minutes * 60 + seconds
    else:
        duration = None

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
    ffmpeg_bin: str,
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

    cmd: List[str] = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
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
    ffmpeg_bin, ffprobe_bin = _resolve_ffmpeg_binaries()

    path = Path(video_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    source_width, source_height, duration = _probe_video(path, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)
    start_sec, end_sec = _validated_time_window(start_sec, end_sec, duration)

    effective_fps = float(fps)
    if max_frames is not None and max_frames > 0 and end_sec is not None and end_sec > start_sec:
        window_duration = float(end_sec - start_sec)
        target_fps = max_frames / max(window_duration, 1e-6)
        effective_fps = min(float(fps), float(target_fps))
        effective_fps = max(effective_fps, 1e-4)

    if end_sec is not None and math.isclose(start_sec, end_sec):
        return []

    cmd, out_width, out_height = _build_ffmpeg_cmd(
        video_path=path,
        ffmpeg_bin=ffmpeg_bin,
        fps=effective_fps,
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
        ts = start_sec + (idx / effective_fps)
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
