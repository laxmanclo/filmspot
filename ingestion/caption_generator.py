from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class CaptionGenerationError(RuntimeError):
    """Raised when caption generation fails."""


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class BLIPCaptionGenerator:
    """BLIP/BLIP-2 caption generator wrapper built on Hugging Face Transformers."""

    model_name: str = "Salesforce/blip2-opt-2.7b"
    device: str = "cpu"
    torch_dtype: torch.dtype = torch.float32
    processor: object = field(init=False, repr=False)
    model: torch.nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        if self.device.startswith("cuda"):
            # For larger BLIP-2 models this helps avoid manual placement.
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=self.torch_dtype,
                device_map="auto",
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=self.torch_dtype,
            )
            self.model.to(self.device)

        self.model.eval()

    @property
    def model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(
        self,
        frames: Sequence[Tuple[float, Image.Image]],
        batch_size: int = 4,
        max_new_tokens: int = 30,
        prompt: str | None = None,
    ) -> List[Tuple[float, str]]:
        """Generate captions for timestamped frames."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if not frames:
            return []

        prompt_text = (prompt or "").strip()
        out: List[Tuple[float, str]] = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            ts_batch = [ts for ts, _ in batch]
            image_batch = [img.convert("RGB") for _, img in batch]

            processor_kwargs = {
                "images": image_batch,
                "return_tensors": "pt",
                "padding": True,
            }
            if prompt_text:
                processor_kwargs["text"] = [prompt_text] * len(batch)

            inputs = self.processor(**processor_kwargs)
            inputs = {k: v.to(self.model_device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            for ts, caption in zip(ts_batch, captions):
                out.append((float(ts), " ".join(caption.split())))

        return out


@lru_cache(maxsize=4)
def _cached_captioner(model_name: str, device: str) -> BLIPCaptionGenerator:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return BLIPCaptionGenerator(model_name=model_name, device=device, torch_dtype=dtype)


def generate_captions(
    frames: Sequence[Tuple[float, Image.Image]],
    model_name: str = "Salesforce/blip2-opt-2.7b",
    device: str | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 30,
    prompt: str | None = None,
) -> List[Tuple[float, str]]:
    """
    Generate BLIP captions for timestamped frames.

    Args:
        frames: Sequence of `(timestamp_sec, PIL.Image)` tuples.
        model_name: Hugging Face BLIP/BLIP-2 model ID.
        device: Target torch device, auto-selects CUDA if available else CPU.
        batch_size: Number of frames per inference batch.
        max_new_tokens: Maximum generated caption length.
        prompt: Optional prompt prefix passed to the model.

    Returns:
        List of `(timestamp_sec, caption)` tuples.
    """
    chosen_device = device or _default_device()

    try:
        captioner = _cached_captioner(model_name=model_name, device=chosen_device)
        return captioner.generate(
            frames=frames,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
        )
    except Exception as exc:  # pragma: no cover - keeps API-level errors consistent
        raise CaptionGenerationError(
            f"Caption generation failed for model '{model_name}' on device '{chosen_device}': {exc}"
        ) from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate captions for a single image using BLIP/BLIP-2")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b", help="Model id")
    parser.add_argument("--device", type=str, default=None, help="Torch device: cpu/cuda")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--prompt", type=str, default="")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    captions = generate_captions(
        frames=[(0.0, img)],
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt or None,
    )
    print(captions[0][1] if captions else "")
