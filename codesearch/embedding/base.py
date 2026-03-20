from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return an embedding vector for each input text."""

    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding vector dimensions for this provider."""
