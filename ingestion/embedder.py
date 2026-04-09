from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

import clip
import numpy as np
import torch
from PIL import Image

from vector_store import ChromaEmbeddingStore


class EmbeddingError(RuntimeError):
    """Raised when CLIP embedding operations fail."""


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vecs / norms


@lru_cache(maxsize=4)
def _load_clip(model_name: str, device: str):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


class CLIPEmbedder:
    """CLIP image/text embedder for frame and query encoding."""

    def __init__(self, model_name: str = "ViT-B/32", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or _default_device()
        self.model, self.preprocess = _load_clip(model_name=self.model_name, device=self.device)

    def encode_frames(self, frames: Sequence[Tuple[float, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """Encode timestamped frames to normalized CLIP image embeddings of shape (N, D)."""
        if not frames:
            return np.zeros((0, 512), dtype=np.float32)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        tensors = [self.preprocess(img.convert("RGB")) for _, img in frames]
        outputs: list[np.ndarray] = []

        with torch.inference_mode():
            for i in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[i : i + batch_size]).to(self.device)
                emb = self.model.encode_image(batch)
                outputs.append(emb.detach().cpu().numpy().astype(np.float32))

        arr = np.concatenate(outputs, axis=0)
        return _l2_normalize(arr)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query into a normalized CLIP embedding of shape (D,)."""
        normalized = " ".join(text.split()).strip()
        if not normalized:
            raise ValueError("text must not be empty")

        tokens = clip.tokenize([normalized]).to(self.device)
        with torch.inference_mode():
            emb = self.model.encode_text(tokens)

        arr = emb.detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(arr)[0]

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a query image into a normalized CLIP embedding of shape (D,)."""
        image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            emb = self.model.encode_image(image_tensor)

        arr = emb.detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(arr)[0]


def encode_frames(
    frames: Sequence[Tuple[float, Image.Image]],
    model_name: str = "ViT-B/32",
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    return CLIPEmbedder(model_name=model_name, device=device).encode_frames(frames=frames, batch_size=batch_size)


def encode_text(text: str, model_name: str = "ViT-B/32", device: str | None = None) -> np.ndarray:
    return CLIPEmbedder(model_name=model_name, device=device).encode_text(text=text)


def encode_image(image: Image.Image, model_name: str = "ViT-B/32", device: str | None = None) -> np.ndarray:
    return CLIPEmbedder(model_name=model_name, device=device).encode_image(image=image)


def save_embeddings(embeddings: np.ndarray, file_path: str | Path) -> Path:
    """Save embeddings to a local .npy file."""
    path = Path(file_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32))
    return path


def load_embeddings(file_path: str | Path) -> np.ndarray:
    """Restore embeddings from a local .npy file."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    emb = np.load(path)
    if emb.ndim != 2:
        raise ValueError("Expected embeddings with shape (N, D)")
    return emb.astype(np.float32)


def save_embeddings_to_chroma(
    movie_id: str,
    embeddings: np.ndarray,
    timestamps: Sequence[float],
    captions: Sequence[str] | None = None,
    transcripts: Sequence[str] | None = None,
    node_ids: Sequence[int] | None = None,
    persist_directory: str | Path = "chroma_db",
) -> list[str]:
    """Persist frame embeddings and metadata into ChromaDB."""
    store = ChromaEmbeddingStore(persist_directory=persist_directory)
    return store.upsert_embeddings(
        movie_id=movie_id,
        embeddings=embeddings,
        timestamps=timestamps,
        captions=captions,
        transcripts=transcripts,
        node_ids=node_ids,
    )


def restore_embeddings_from_chroma(
    movie_id: str,
    persist_directory: str | Path = "chroma_db",
) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
    """Restore all embeddings and metadata for a movie from ChromaDB."""
    store = ChromaEmbeddingStore(persist_directory=persist_directory)
    ids, emb, metadatas = store.restore_embeddings(movie_id=movie_id)
    return ids, emb, metadatas


if __name__ == "__main__":
    import argparse

    from ingestion.frame_extractor import extract_frames

    parser = argparse.ArgumentParser(description="Encode video frames with CLIP and persist embeddings")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--movie-id", type=str, default="sample_movie")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--chroma-dir", type=str, default="chroma_db")
    parser.add_argument("--save-npy", type=str, default="data/index/sample_movie/frame_embeddings.npy")
    args = parser.parse_args()

    frame_data = extract_frames(args.video, fps=args.fps, max_frames=args.max_frames)
    if not frame_data:
        raise SystemExit("No frames extracted; cannot embed.")

    embeddings = encode_frames(frame_data)
    timestamps = [t for t, _ in frame_data]

    npy_path = save_embeddings(embeddings, args.save_npy)
    vec_ids = save_embeddings_to_chroma(
        movie_id=args.movie_id,
        embeddings=embeddings,
        timestamps=timestamps,
        persist_directory=args.chroma_dir,
    )

    restored_ids, restored_emb, _ = restore_embeddings_from_chroma(
        movie_id=args.movie_id,
        persist_directory=args.chroma_dir,
    )

    print(f"Extracted frames: {len(frame_data)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved npy: {npy_path}")
    print(f"Upserted vectors: {len(vec_ids)}")
    print(f"Restored vectors: {len(restored_ids)} with shape {restored_emb.shape}")
