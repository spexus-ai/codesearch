from __future__ import annotations

import json
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal runtime environments
    httpx = None  # type: ignore[assignment]

from codesearch.errors import ProviderError
from codesearch.embedding.base import EmbeddingProvider


class OpenAIProvider(EmbeddingProvider):
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        if not api_key:
            raise ProviderError("API key is required for OpenAIProvider")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._dimensions: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), 100):
            embeddings.extend(self._embed_batch(texts[start : start + 100]))
        return embeddings

    def dimensions(self) -> int:
        if self._dimensions is None:
            raise ProviderError("OpenAI embedding dimensions are unknown until embed() succeeds")
        return self._dimensions

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = self._request_embeddings(texts)
        data = payload.get("data")
        if not isinstance(data, list):
            raise ProviderError("OpenAI response did not contain a data list")

        embeddings = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise ProviderError("OpenAI response item did not contain an embedding list")
            embeddings.append([float(value) for value in embedding])

        if len(embeddings) != len(texts):
            raise ProviderError("OpenAI returned an unexpected number of embeddings")
        if embeddings:
            dimensions = len(embeddings[0])
            if self._dimensions is None:
                self._dimensions = dimensions
            elif self._dimensions != dimensions:
                raise ProviderError(f"OpenAI embedding dimensions changed from {self._dimensions} to {dimensions}")
        return embeddings

    def _request_embeddings(self, texts: list[str]) -> dict:
        if httpx is not None:
            return self._request_embeddings_httpx(texts)
        return self._request_embeddings_urllib(texts)

    def _request_embeddings_httpx(self, texts: list[str]) -> dict:
        try:
            with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
                response = client.post(
                    "/embeddings",
                    json={"model": self.model, "input": texts},
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise ProviderError("Invalid API key") from exc
            raise ProviderError(f"OpenAI embedding request failed: {exc.response.status_code} {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI embedding request failed: {exc}") from exc

        return response.json()

    def _request_embeddings_urllib(self, texts: list[str]) -> dict:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib_request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=30.0) as response:
                raw_body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401:
                raise ProviderError("Invalid API key") from exc
            raise ProviderError(f"OpenAI embedding request failed: {exc.code} {error_body}") from exc
        except urllib_error.URLError as exc:
            raise ProviderError(f"OpenAI embedding request failed: {exc.reason}") from exc

        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ProviderError("OpenAI response was not valid JSON") from exc
