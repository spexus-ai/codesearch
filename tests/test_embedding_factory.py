import pytest

from codesearch.config import EmbeddingConfig
from codesearch.embedding.base import EmbeddingProvider
from codesearch.embedding.factory import create_provider
import codesearch.embedding.factory as factory_module


class DummyProvider(EmbeddingProvider):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def dimensions(self) -> int:
        return 1


def test_embedding_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_factory_creates_onnx_direct_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_module, "OnnxDirectProvider", DummyProvider)

    provider = create_provider(
        EmbeddingConfig(provider="onnx", model="all-MiniLM-L6-v2")
    )

    assert isinstance(provider, DummyProvider)
    assert provider.kwargs == {"model": "all-MiniLM-L6-v2"}


def test_factory_creates_sentence_transformers_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_module, "SentenceTransformersProvider", DummyProvider)

    provider = create_provider(
        EmbeddingConfig(provider="sentence-transformers", model="all-MiniLM-L6-v2", backend="onnx")
    )

    assert isinstance(provider, DummyProvider)
    assert provider.kwargs == {"model": "all-MiniLM-L6-v2", "backend": "onnx"}


def test_factory_creates_ollama_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_module, "OllamaProvider", DummyProvider)

    provider = create_provider(
        EmbeddingConfig(provider="ollama", model="qwen3-embedding:0.6b", base_url="http://localhost:11434")
    )

    assert isinstance(provider, DummyProvider)
    assert provider.kwargs == {"model": "qwen3-embedding:0.6b", "base_url": "http://localhost:11434"}


def test_factory_creates_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_module, "OpenAIProvider", DummyProvider)

    provider = create_provider(
        EmbeddingConfig(provider="openai", model="text-embedding-3-small", api_key="secret")
    )

    assert isinstance(provider, DummyProvider)
    assert provider.kwargs == {"model": "text-embedding-3-small", "api_key": "secret"}


def test_factory_creates_openai_compatible_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_module, "OpenAICompatibleProvider", DummyProvider)

    provider = create_provider(
        EmbeddingConfig(
            provider="openai-compatible",
            model="voyage-code-3",
            api_key="secret",
            base_url="https://example.test/v1",
        )
    )

    assert isinstance(provider, DummyProvider)
    assert provider.kwargs == {
        "model": "voyage-code-3",
        "api_key": "secret",
        "base_url": "https://example.test/v1",
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EmbeddingConfig(provider="openai", model="x"), "embedding.api_key is required for provider openai"),
        (
            EmbeddingConfig(provider="openai-compatible", model="x", api_key="secret"),
            "embedding.base_url is required for provider openai-compatible",
        ),
        (EmbeddingConfig(provider="unknown", model="x"), "Unsupported embedding provider: unknown"),
    ],
)
def test_factory_rejects_invalid_config(config: EmbeddingConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_provider(config)
