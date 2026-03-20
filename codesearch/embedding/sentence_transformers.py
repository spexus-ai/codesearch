from __future__ import annotations

import importlib

from codesearch.errors import ProviderError
from codesearch.embedding.base import EmbeddingProvider


class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self, model: str = "codesage/codesage-small", backend: str = "onnx"):
        if backend not in {"onnx", "torch"}:
            raise ValueError(f"Unsupported sentence-transformers backend: {backend}")
        self.model = model or "codesage/codesage-small"
        self.backend = backend
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=max(1, min(32, len(texts))),
                show_progress_bar=False,
            )
        except Exception as exc:
            raise ProviderError(f"Failed to embed texts with sentence-transformers model {self.model}: {exc}") from exc
        return self._normalize_embeddings(embeddings)

    def dimensions(self) -> int:
        model = self._get_model()
        get_dimensions = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(get_dimensions):
            value = get_dimensions()
            if value is not None:
                return int(value)
        config = getattr(model, "_model_config", None) or getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
        raise ProviderError(f"Could not determine embedding dimensions for sentence-transformers model {self.model}")

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            raise ProviderError(
                "sentence-transformers is not installed. Install the project dependencies to use local embeddings."
            ) from exc

        try:
            model_kwargs = {}
            if self.backend == "onnx":
                model_kwargs["file_name"] = "onnx/model.onnx"
            self._model = sentence_transformers.SentenceTransformer(
                self.model, backend=self.backend, trust_remote_code=True, model_kwargs=model_kwargs,
            )
        except Exception as exc:
            raise ProviderError(f"Failed to load sentence-transformers model {self.model}: {exc}") from exc
        return self._model

    def _normalize_embeddings(self, embeddings) -> list[list[float]]:
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        normalized: list[list[float]] = []
        for embedding in embeddings:
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            normalized.append([float(value) for value in embedding])
        return normalized
