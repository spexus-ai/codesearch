from __future__ import annotations

from codesearch.config import EmbeddingConfig
from codesearch.embedding.base import EmbeddingProvider
from codesearch.embedding.ollama import OllamaProvider
from codesearch.embedding.openai import OpenAIProvider
from codesearch.embedding.openai_compatible import OpenAICompatibleProvider
from codesearch.embedding.sentence_transformers import SentenceTransformersProvider


def create_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    match config.provider:
        case "sentence-transformers":
            return SentenceTransformersProvider(model=config.model, backend=config.backend)
        case "ollama":
            return OllamaProvider(model=config.model, base_url=config.base_url or "http://localhost:11434")
        case "openai":
            if not config.api_key:
                raise ValueError("embedding.api_key is required for provider openai")
            return OpenAIProvider(model=config.model, api_key=config.api_key)
        case "openai-compatible" | "openai_compatible":
            if not config.api_key:
                raise ValueError("embedding.api_key is required for provider openai-compatible")
            if not config.base_url:
                raise ValueError("embedding.base_url is required for provider openai-compatible")
            return OpenAICompatibleProvider(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
            )
        case _:
            raise ValueError(f"Unsupported embedding provider: {config.provider}")
