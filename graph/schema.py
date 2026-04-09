from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class SentenceNode:
    """
    One per-second multimodal unit used in the temporal sentence graph.

    Spec fields (Phase 2):
      - t: timestamp in seconds
      - image_emb: CLIP image embedding vector
      - caption: BLIP-generated frame caption
      - transcript: Whisper transcript text aligned to this second
    """

    t: float
    image_emb: np.ndarray
    caption: str
    transcript: str

    def __post_init__(self) -> None:
        self.t = float(self.t)
        self.caption = str(self.caption)
        self.transcript = str(self.transcript)

        emb = np.asarray(self.image_emb, dtype=np.float32)
        if emb.ndim != 1:
            raise ValueError("SentenceNode.image_emb must be a 1D embedding vector")
        self.image_emb = emb

    @property
    def second(self) -> int:
        """Integer second bucket for this node."""
        return int(self.t)

    def text_for_bm25(self) -> str:
        """Concatenated text used by transcript/caption lexical retrieval."""
        return f"{self.caption} {self.transcript}".strip()

    def to_metadata(self) -> dict[str, Any]:
        """Serialize lightweight metadata (without embedding tensor) for storage/indexing."""
        return {
            "t": self.t,
            "caption": self.caption,
            "transcript": self.transcript,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], image_emb: np.ndarray) -> "SentenceNode":
        """Reconstruct node from stored metadata + embedding vector."""
        return cls(
            t=float(metadata.get("t", 0.0)),
            image_emb=image_emb,
            caption=str(metadata.get("caption", "")),
            transcript=str(metadata.get("transcript", "")),
        )
