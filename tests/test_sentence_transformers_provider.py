import importlib.util
import os
from types import SimpleNamespace

import pytest

from codesearch.embedding.sentence_transformers import SentenceTransformersProvider
from codesearch.errors import ProviderError


LIVE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LIVE_MODEL_ENV = "CODESEARCH_RUN_LIVE_MODEL_TESTS"


class FakeSentenceTransformerModel:
    instances = 0

    def __init__(self, model_name: str, backend: str, trust_remote_code: bool = False, model_kwargs: dict | None = None):
        self.model_name = model_name
        self.backend = backend
        self.encode_calls = []
        FakeSentenceTransformerModel.instances += 1

    def encode(self, texts: list[str], batch_size: int, show_progress_bar: bool):
        self.encode_calls.append((list(texts), batch_size, show_progress_bar))
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

    def get_sentence_embedding_dimension(self) -> int:
        return 2


def test_sentence_transformers_provider_lazy_loads_and_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SentenceTransformersProvider(model="test-model", backend="onnx")
    module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformerModel)
    monkeypatch.setattr("codesearch.embedding.sentence_transformers.importlib.import_module", lambda _: module)
    FakeSentenceTransformerModel.instances = 0

    first = provider.embed(["alpha", "beta"])
    second = provider.embed(["gamma"])

    assert FakeSentenceTransformerModel.instances == 1
    assert first == [[5.0, 0.0], [4.0, 1.0]]
    assert second == [[5.0, 0.0]]
    assert provider.dimensions() == 2


def test_sentence_transformers_provider_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported sentence-transformers backend: invalid"):
        SentenceTransformersProvider(model="test-model", backend="invalid")


def test_sentence_transformers_provider_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SentenceTransformersProvider(model="test-model")

    def _raise(_: str):
        raise ModuleNotFoundError("sentence_transformers")

    monkeypatch.setattr("codesearch.embedding.sentence_transformers.importlib.import_module", _raise)

    with pytest.raises(ProviderError, match="sentence-transformers is not installed"):
        provider.dimensions()


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None, reason="sentence-transformers is not installed")
def test_sentence_transformers_provider_integration_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    live_download_enabled = os.environ.get(LIVE_MODEL_ENV) == "1"
    if not live_download_enabled:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    provider = SentenceTransformersProvider(model=LIVE_MODEL, backend="torch")

    try:
        embeddings = provider.embed(["hello world", "semantic search"])
    except ProviderError as exc:
        if live_download_enabled:
            raise
        pytest.skip(f"{LIVE_MODEL} is not fully cached locally; set {LIVE_MODEL_ENV}=1 to run the live download test: {exc}")

    assert len(embeddings) == 2
    assert provider.dimensions() == len(embeddings[0])
