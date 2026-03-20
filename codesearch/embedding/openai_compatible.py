from __future__ import annotations

from codesearch.embedding.openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    def __init__(self, model: str, api_key: str, base_url: str):
        super().__init__(model=model, api_key=api_key, base_url=base_url)
