from __future__ import annotations

import importlib

from codesearch.errors import ProviderError
from codesearch.embedding.base import EmbeddingProvider


class _LazyHttpx:
    def __init__(self) -> None:
        object.__setattr__(self, "_module", None)
        object.__setattr__(self, "_overrides", {})

    def _load_module(self):
        module = object.__getattribute__(self, "_module")
        if module is None:
            module = importlib.import_module("httpx")
            object.__setattr__(self, "_module", module)
        return module

    def __getattr__(self, name: str):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(self._load_module(), name)

    def __setattr__(self, name: str, value) -> None:
        if name in {"_module", "_overrides"}:
            object.__setattr__(self, name, value)
            return
        object.__getattribute__(self, "_overrides")[name] = value

    def __delattr__(self, name: str) -> None:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            del overrides[name]
            return
        raise AttributeError(name)


httpx = _LazyHttpx()


class OllamaProvider(EmbeddingProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimensions: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def dimensions(self) -> int:
        if self._dimensions is None:
            raise ProviderError("Ollama embedding dimensions are unknown until embed() succeeds")
        return self._dimensions

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            httpx_client = httpx.Client
        except ModuleNotFoundError as exc:
            raise ProviderError("httpx is required for Ollama provider. Install: pip install httpx") from exc

        try:
            with httpx_client(base_url=self.base_url, timeout=30.0) as client:
                response = client.post("/api/embed", json={"model": self.model, "input": texts})
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ProviderError(f"Cannot connect to Ollama at {self.base_url}") from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Ollama embedding request failed: {exc.response.status_code} {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama embedding request failed: {exc}") from exc

        payload = response.json()
        raw_embeddings = payload.get("embeddings")
        if not isinstance(raw_embeddings, list):
            raise ProviderError("Ollama response did not contain an embeddings list")

        embeddings = [[float(value) for value in embedding] for embedding in raw_embeddings]
        if len(embeddings) != len(texts):
            raise ProviderError("Ollama returned an unexpected number of embeddings")
        if embeddings:
            batch_dimensions = len(embeddings[0])
            if self._dimensions is None:
                self._dimensions = batch_dimensions
            elif self._dimensions != batch_dimensions:
                raise ProviderError(
                    f"Ollama embedding dimensions changed from {self._dimensions} to {batch_dimensions}"
                )
        return embeddings
