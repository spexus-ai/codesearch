import httpx
import pytest

from codesearch.embedding.openai import OpenAIProvider
from codesearch.embedding.openai_compatible import OpenAICompatibleProvider
from codesearch.errors import ProviderError


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
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

    def post(self, path: str, json: dict, headers: dict):
        FakeClient.calls.append(
            {
                "base_url": self.base_url,
                "path": path,
                "json": json,
                "headers": headers,
            }
        )
        data = [{"embedding": [float(len(text)), float(index)]} for index, text in enumerate(json["input"])]
        return FakeResponse({"data": data})


def test_openai_provider_batches_requests_and_caches_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(model="text-embedding-3-small", api_key="secret")
    FakeClient.calls = []
    monkeypatch.setattr("codesearch.embedding.openai.httpx.Client", FakeClient)

    embeddings = provider.embed([f"text-{index}" for index in range(101)])

    assert len(embeddings) == 101
    assert len(FakeClient.calls) == 2
    assert len(FakeClient.calls[0]["json"]["input"]) == 100
    assert len(FakeClient.calls[1]["json"]["input"]) == 1
    assert FakeClient.calls[0]["path"] == "/embeddings"
    assert FakeClient.calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert provider.dimensions() == 2


def test_openai_provider_requires_api_key() -> None:
    with pytest.raises(ProviderError, match="API key is required for OpenAIProvider"):
        OpenAIProvider(model="text-embedding-3-small", api_key="")


def test_openai_provider_reports_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(model="text-embedding-3-small", api_key="secret")

    class FailingClient:
        def __init__(self, *, base_url: str, timeout: float):
            self.base_url = base_url
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, path: str, json: dict, headers: dict):
            return FakeResponse({}, status_code=401, text="unauthorized")

    monkeypatch.setattr("codesearch.embedding.openai.httpx.Client", FailingClient)

    with pytest.raises(ProviderError, match="Invalid API key"):
        provider.embed(["hello"])


def test_openai_provider_requires_successful_embed_for_dimensions() -> None:
    provider = OpenAIProvider(model="text-embedding-3-small", api_key="secret")

    with pytest.raises(ProviderError, match="dimensions are unknown until embed\\(\\) succeeds"):
        provider.dimensions()


def test_openai_compatible_provider_uses_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatibleProvider(
        model="voyage-code-3",
        api_key="secret",
        base_url="https://openrouter.ai/api/v1/",
    )
    FakeClient.calls = []
    monkeypatch.setattr("codesearch.embedding.openai.httpx.Client", FakeClient)

    provider.embed(["hello"])

    assert FakeClient.calls[0]["base_url"] == "https://openrouter.ai/api/v1"


def test_openai_provider_falls_back_to_urllib_when_httpx_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatibleProvider(
        model="mock-embeddings",
        api_key="secret",
        base_url="http://localhost:8080/v1/",
    )

    class FakeUrlopenResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{\"data\": [{\"embedding\": [1.0, 2.0]}]}'

    def fake_urlopen(request, timeout: float):
        assert request.full_url == "http://localhost:8080/v1/embeddings"
        assert request.get_header("Authorization") == "Bearer secret"
        assert timeout == 30.0
        return FakeUrlopenResponse()

    monkeypatch.setattr("codesearch.embedding.openai.httpx", None)
    monkeypatch.setattr("codesearch.embedding.openai.urllib_request.urlopen", fake_urlopen)

    assert provider.embed(["hello"]) == [[1.0, 2.0]]
    assert provider.dimensions() == 2
