import os
from types import SimpleNamespace

import pytest

from codesearch.embedding.onnx_direct import OnnxDirectProvider
from codesearch.errors import ProviderError


LIVE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LIVE_MODEL_ENV = "CODESEARCH_RUN_LIVE_MODEL_TESTS"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEncoding:
    def __init__(self, ids, attention_mask, type_ids):
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids


class FakeTokenizer:
    def __init__(self):
        self._pad_enabled = False
        self._trunc_enabled = False

    def enable_padding(self, **_kwargs):
        self._pad_enabled = True

    def enable_truncation(self, **_kwargs):
        self._trunc_enabled = True

    def encode_batch(self, texts):
        max_len = max(len(t.split()) + 2 for t in texts)  # simulate tokens
        result = []
        for text in texts:
            n_tokens = len(text.split()) + 2  # [CLS] + words + [SEP]
            ids = list(range(n_tokens)) + [0] * (max_len - n_tokens)
            mask = [1] * n_tokens + [0] * (max_len - n_tokens)
            type_ids = [0] * max_len
            result.append(FakeEncoding(ids, mask, type_ids))
        return result


class FakeInput:
    def __init__(self, name):
        self.name = name


class FakeOutput:
    def __init__(self, shape):
        self.shape = shape


class FakeSession:
    def __init__(self):
        import numpy as np
        self._np = np

    def get_inputs(self):
        return [FakeInput("input_ids"), FakeInput("attention_mask"), FakeInput("token_type_ids")]

    def get_outputs(self):
        return [FakeOutput([None, None, 4])]

    def run(self, _output_names, feeds):
        np = self._np
        batch, seq_len = feeds["input_ids"].shape
        # Return constant token embeddings so mean pooling gives predictable results
        return [np.ones((batch, seq_len, 4), dtype=np.float32)]


# ---------------------------------------------------------------------------
# Unit tests (fully mocked, no model download)
# ---------------------------------------------------------------------------


def _patch_provider(monkeypatch, provider):
    """Patch the provider to use fakes instead of real libraries."""
    import numpy as np

    def fake_hf_hub_download(repo, filename, **_kwargs):
        return f"/fake/{repo}/{filename}"

    fake_hf = SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    fake_ort = SimpleNamespace(
        InferenceSession=lambda path, providers=None: FakeSession()
    )
    fake_tokenizers = SimpleNamespace(
        Tokenizer=SimpleNamespace(from_file=lambda path: FakeTokenizer())
    )

    import builtins
    real_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            return fake_hf
        if name == "onnxruntime":
            return fake_ort
        if name == "tokenizers":
            return fake_tokenizers
        if name == "numpy":
            return np
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)


def test_onnx_direct_provider_embeds_texts(monkeypatch) -> None:
    provider = OnnxDirectProvider(model="test-model")
    _patch_provider(monkeypatch, provider)

    results = provider.embed(["hello world", "foo"])

    assert len(results) == 2
    assert len(results[0]) == 4
    # Constant 1.0 token embeddings → mean pooling gives [1,1,1,1] → L2-normalized to [0.5,0.5,0.5,0.5]
    expected = 1.0 / (4 ** 0.5)
    assert all(abs(v - expected) < 1e-5 for v in results[0])


def test_onnx_direct_provider_empty_input() -> None:
    provider = OnnxDirectProvider(model="test-model")
    assert provider.embed([]) == []


def test_onnx_direct_provider_normalizes_short_model_name() -> None:
    provider = OnnxDirectProvider(model="all-MiniLM-L6-v2")
    assert provider.model == "sentence-transformers/all-MiniLM-L6-v2"


def test_onnx_direct_provider_keeps_full_model_name() -> None:
    provider = OnnxDirectProvider(model="org/custom-model")
    assert provider.model == "org/custom-model"


def test_onnx_direct_provider_dimensions_from_output_shape(monkeypatch) -> None:
    provider = OnnxDirectProvider(model="test-model")
    _patch_provider(monkeypatch, provider)

    dims = provider.dimensions()

    assert dims == 4


def test_onnx_direct_provider_lazy_loads_once(monkeypatch) -> None:
    provider = OnnxDirectProvider(model="test-model")
    _patch_provider(monkeypatch, provider)

    provider.embed(["first"])
    session1 = provider._session
    provider.embed(["second"])
    session2 = provider._session

    assert session1 is session2


def test_onnx_direct_provider_reports_missing_onnxruntime(monkeypatch) -> None:
    provider = OnnxDirectProvider(model="test-model")

    import builtins
    real_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            return SimpleNamespace(hf_hub_download=lambda *a, **k: "/fake")
        if name == "onnxruntime":
            raise ModuleNotFoundError("onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)

    with pytest.raises(ProviderError, match="onnxruntime is required"):
        provider.embed(["test"])


def test_onnx_direct_provider_reports_missing_tokenizers(monkeypatch) -> None:
    provider = OnnxDirectProvider(model="test-model")

    import builtins
    real_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            return SimpleNamespace(hf_hub_download=lambda *a, **k: "/fake")
        if name == "onnxruntime":
            return SimpleNamespace(InferenceSession=lambda *a, **k: FakeSession())
        if name == "tokenizers":
            raise ModuleNotFoundError("tokenizers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)

    with pytest.raises(ProviderError, match="tokenizers is required"):
        provider.embed(["test"])


# ---------------------------------------------------------------------------
# Integration test (requires cached model)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get(LIVE_MODEL_ENV) != "1",
    reason=f"Set {LIVE_MODEL_ENV}=1 to run live model tests",
)
def test_onnx_direct_provider_integration_real_model() -> None:
    provider = OnnxDirectProvider(model=LIVE_MODEL)

    embeddings = provider.embed(["hello world", "semantic search"])

    assert len(embeddings) == 2
    assert provider.dimensions() == len(embeddings[0])
    assert provider.dimensions() == 384
