from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class QueryDecompositionError(RuntimeError):
    """Raised when query decomposition cannot be completed."""


@dataclass(slots=True)
class DecomposedQuery:
    visual: str
    dialogue: str
    source: str = "heuristic"

    def as_dict(self) -> dict[str, str]:
        return {"visual": self.visual, "dialogue": self.dialogue}


class QueryDecomposer:
    """Gemini-backed text query decomposition into visual and dialogue sub-queries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 220,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split()).strip()

    def _fallback(self, text: str) -> DecomposedQuery:
        normalized = self._normalize_text(text)
        if not normalized:
            raise ValueError("text query must not be empty")
        return DecomposedQuery(visual=normalized, dialogue=normalized, source="heuristic")

    @staticmethod
    def _extract_json_payload(raw: str) -> dict[str, Any]:
        match = _JSON_OBJECT_PATTERN.search(raw)
        if not match:
            raise QueryDecompositionError("Gemini response did not contain a JSON object")

        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise QueryDecompositionError("Gemini response JSON payload must be an object")
        return payload

    def _call_gemini(self, text: str) -> DecomposedQuery:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise QueryDecompositionError("google-genai package is not available") from exc

        if not self.api_key:
            raise QueryDecompositionError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")

        client = genai.Client(api_key=self.api_key)

        prompt = (
            "Decompose the user query for multimodal video retrieval. "
            "Return strict JSON with exactly two string keys: "
            '{"visual": "...", "dialogue": "..."}. '
            "visual focuses on visible actions/objects/scenes. "
            "dialogue focuses on spoken words/phrases likely in transcript. "
            "Do not include markdown or extra keys."
        )

        resp = client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                f"User query: {text}",
                "Return JSON only.",
            ],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )

        raw = str(getattr(resp, "text", "") or "").strip()
        payload = self._extract_json_payload(raw)

        visual = self._normalize_text(str(payload.get("visual", "")))
        dialogue = self._normalize_text(str(payload.get("dialogue", "")))

        if not visual:
            visual = self._normalize_text(text)
        if not dialogue:
            dialogue = self._normalize_text(text)

        return DecomposedQuery(visual=visual, dialogue=dialogue, source="gemini")

    def decompose(self, text: str) -> DecomposedQuery:
        normalized = self._normalize_text(text)
        if not normalized:
            raise ValueError("text query must not be empty")

        try:
            return self._call_gemini(normalized)
        except Exception:
            return self._fallback(normalized)


def decompose_query(text: str) -> dict[str, str]:
    """Convenience wrapper returning the phase-spec JSON shape."""
    return QueryDecomposer().decompose(text).as_dict()
