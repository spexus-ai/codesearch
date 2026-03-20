from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codesearch.embedding.base import EmbeddingProvider
from codesearch.errors import CodeSearchError, DimensionMismatchError, ProviderError, StorageError
from codesearch.storage import Storage


@dataclass(slots=True)
class SearchResult:
    repo: str
    path: str
    line: int
    score: float
    snippet: str
    lang: str


class Searcher:
    def __init__(self, storage: Storage, provider: EmbeddingProvider):
        self.storage = storage
        self.provider = provider

    def search(
        self,
        query: str,
        repo: str | None = None,
        langs: list[str] | None = None,
        path_glob: str | None = None,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise CodeSearchError("Query cannot be empty")

        repo_id = self._resolve_repo_id(repo)
        index_dimensions = self.storage.get_meta("embedding_dimensions")
        if index_dimensions is not None:
            self._validate_provider_dimensions(int(index_dimensions))

        embeddings = self.provider.embed([normalized_query])
        if len(embeddings) != 1:
            raise StorageError("Embedding provider returned an unexpected number of query vectors")

        query_embedding = [float(value) for value in embeddings[0]]
        if index_dimensions is not None and len(query_embedding) != int(index_dimensions):
            raise DimensionMismatchError(
                f"Index dimensions {index_dimensions} do not match query dimensions {len(query_embedding)}. "
                "Run index --full."
            )

        results = self.storage.search(
            query_embedding,
            limit=limit,
            threshold=threshold,
            repo_id=repo_id,
            langs=langs,
            path_glob=path_glob,
        )
        return [
            SearchResult(
                repo=result.repo,
                path=result.path,
                line=result.line,
                score=result.score,
                snippet=result.snippet,
                lang=result.lang,
            )
            for result in results
        ]

    def _resolve_repo_id(self, repo: str | None) -> int | None:
        if repo is None:
            return None

        repo_text = repo.strip()
        if not repo_text:
            raise CodeSearchError("Repository filter cannot be empty")

        repo_path = str(Path(repo_text).expanduser().resolve())
        for item in self.storage.list_repos():
            if item.name == repo_text or item.path == repo_path:
                return item.id
        raise StorageError(f"Repository not found: {repo_text}")

    def _validate_provider_dimensions(self, index_dimensions: int) -> None:
        try:
            provider_dimensions = self.provider.dimensions()
        except ProviderError:
            return
        if provider_dimensions != index_dimensions:
            raise DimensionMismatchError(
                f"Index dimensions {index_dimensions} do not match query dimensions {provider_dimensions}. "
                "Run index --full."
            )
