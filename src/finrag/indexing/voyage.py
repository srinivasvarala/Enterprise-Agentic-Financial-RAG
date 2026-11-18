"""Voyage dense embedder over plain HTTP (ADR-0005). Model id and API key come from Settings."""

from collections.abc import Sequence

import httpx

from finrag.core.config import Settings
from finrag.core.errors import TransientError


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int,
        base_url: str = "https://api.voyageai.com/v1",
        batch_size: int = 64,
        client: httpx.Client | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "VoyageEmbedder":
        key = settings.voyage_api_key.get_secret_value()
        if not key:
            raise ValueError(
                "FINRAG_VOYAGE_API_KEY is required when FINRAG_EMBEDDER_BACKEND=voyage"
            )
        return cls(
            api_key=key,
            model=settings.voyage_model,
            dimension=settings.voyage_dimension,
            base_url=settings.voyage_base_url,
            batch_size=settings.voyage_batch_size,
            timeout_seconds=settings.voyage_timeout_seconds,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        out: list[tuple[float, ...]] = []
        for i in range(0, len(texts), self._batch_size):
            out += self._embed_batch(list(texts[i : i + self._batch_size]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[tuple[float, ...]]:
        payload = {
            "input": batch,
            "model": self._model,
            "input_type": "document",
            "output_dimension": self._dimension,
        }
        try:
            response = self._client.post("/embeddings", json=payload)
        except httpx.TransportError as exc:  # timeouts, connection failures
            raise TransientError(f"voyage request failed: {exc}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientError(f"voyage returned {response.status_code}")
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda r: r["index"])
        vectors = [tuple(float(x) for x in r["embedding"]) for r in rows]
        if len(vectors) != len(batch) or any(len(v) != self._dimension for v in vectors):
            raise ValueError(
                f"voyage returned unexpected shapes; expected {len(batch)} x {self._dimension}"
            )
        return vectors
