"""Direct ONNX Runtime embedding provider — bypasses sentence-transformers for fast startup."""

from __future__ import annotations

from codesearch.embedding.base import EmbeddingProvider
from codesearch.errors import ProviderError


class OnnxDirectProvider(EmbeddingProvider):
    """Loads an ONNX model and tokenizer directly via onnxruntime + tokenizers."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if "/" not in model:
            model = f"sentence-transformers/{model}"
        self.model = model
        self._session = None
        self._tokenizer = None
        self._dims: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        session, tokenizer = self._ensure_loaded()
        import numpy as np

        encoded = tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

        input_names = {inp.name for inp in session.get_inputs()}
        feeds: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = token_type_ids

        outputs = session.run(None, feeds)
        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)

        # Mean pooling over tokens, masked by attention_mask.
        mask = attention_mask.astype(np.float32)[:, :, np.newaxis]
        pooled = np.sum(token_embeddings * mask, axis=1) / np.maximum(
            np.sum(mask, axis=1), 1e-9
        )

        # L2-normalize to match sentence-transformers output.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-9)

        self._dims = pooled.shape[1]
        return pooled.tolist()

    def dimensions(self) -> int:
        if self._dims is not None:
            return self._dims
        session, _ = self._ensure_loaded()
        output_shape = session.get_outputs()[0].shape
        if len(output_shape) >= 2 and isinstance(output_shape[-1], int):
            self._dims = output_shape[-1]
            return self._dims
        self.embed(["_"])
        return self._dims  # type: ignore[return-value]

    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._session is not None:
            return self._session, self._tokenizer

        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            raise ProviderError(
                "huggingface-hub is required for the onnx provider."
            ) from exc
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ProviderError(
                "onnxruntime is required for the onnx provider."
            ) from exc
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as exc:
            raise ProviderError(
                "tokenizers is required for the onnx provider."
            ) from exc

        try:
            model_path = hf_hub_download(self.model, "onnx/model.onnx")
            tokenizer_path = hf_hub_download(self.model, "tokenizer.json")
        except Exception as exc:
            raise ProviderError(
                f"Failed to locate model files for {self.model}: {exc}"
            ) from exc

        try:
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ProviderError(
                f"Failed to load ONNX model {self.model}: {exc}"
            ) from exc

        try:
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            self._tokenizer.enable_truncation(max_length=512)
        except Exception as exc:
            raise ProviderError(
                f"Failed to load tokenizer for {self.model}: {exc}"
            ) from exc

        return self._session, self._tokenizer
