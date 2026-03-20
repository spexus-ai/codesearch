import os

import httpx
import pytest

from codesearch.embedding.ollama import OllamaProvider
from codesearch.errors import ProviderError


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://localhost:11434/api/embed")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("failure", request=request, response=response)

    def json(self) -> dict:
        return self._payload


class FakeClient:
    calls: list[dict] = []

    def __init__(self, *, base_url: str, timeout: float):
        self.base_url = base_url
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, path: str, json: dict):
        FakeClient.calls.append({"base_url": self.base_url, "path": path, "json": json})
        embeddings = [[float(len(text)), float(index)] for index, text in enumerate(json["input"])]
        return FakeResponse({"embeddings": embeddings})


def test_ollama_provider_batches_requests_and_caches_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(model="nomic-embed-text", base_url="http://localhost:11434/")
    FakeClient.calls = []
    monkeypatch.setattr("codesearch.embedding.ollama.httpx.Client", FakeClient)

    embeddings = provider.embed([f"text-{index}" for index in range(33)])

    assert len(embeddings) == 33
    assert len(FakeClient.calls) == 2
    assert FakeClient.calls[0]["path"] == "/api/embed"
    assert len(FakeClient.calls[0]["json"]["input"]) == 32
    assert len(FakeClient.calls[1]["json"]["input"]) == 1
    assert provider.dimensions() == 2


def test_ollama_provider_reports_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(model="nomic-embed-text")

    class FailingClient:
        def __init__(self, *, base_url: str, timeout: float):
            self.base_url = base_url
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, path: str, json: dict):
            request = httpx.Request("POST", f"{self.base_url}{path}")
            raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr("codesearch.embedding.ollama.httpx.Client", FailingClient)

    with pytest.raises(ProviderError, match="Cannot connect to Ollama at http://localhost:11434"):
        provider.embed(["hello"])


def test_ollama_provider_requires_successful_embed_for_dimensions() -> None:
    provider = OllamaProvider(model="nomic-embed-text")

    with pytest.raises(ProviderError, match="dimensions are unknown until embed\\(\\) succeeds"):
        provider.dimensions()


@pytest.mark.skipif(
    not os.environ.get("CODESEARCH_TEST_OLLAMA_MODEL"),
    reason="CODESEARCH_TEST_OLLAMA_MODEL is not configured",
)
def test_ollama_provider_integration_when_server_is_available() -> None:
    provider = OllamaProvider(
        model=os.environ["CODESEARCH_TEST_OLLAMA_MODEL"],
        base_url=os.environ.get("CODESEARCH_TEST_OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    embeddings = provider.embed(["semantic search", "code retrieval"])

    assert len(embeddings) == 2
    assert provider.dimensions() == len(embeddings[0])
