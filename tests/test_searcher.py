from __future__ import annotations

from pathlib import Path

import pytest

from codesearch.errors import CodeSearchError, DimensionMismatchError, ProviderError, StorageError
from codesearch.searcher import Searcher
from codesearch.storage import Chunk, Storage


class FakeQueryProvider:
    def __init__(self, query_embeddings: dict[str, list[float]], dimensions: int | None = None):
        self.query_embeddings = query_embeddings
        self._dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.query_embeddings[text] for text in texts]

    def dimensions(self) -> int:
        if self._dimensions is None:
            raise ProviderError("dimensions are unknown until embed() succeeds")
        return self._dimensions


def test_searcher_runs_semantic_search_with_filters(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")
    repo_a_path = tmp_path / "repo-a"
    repo_b_path = tmp_path / "repo-b"
    repo_a_path.mkdir()
    repo_b_path.mkdir()
    repo_a_id = storage.add_repo("repo-a", str(repo_a_path))
    repo_b_id = storage.add_repo("repo-b", str(repo_b_path))
    storage.insert_chunks(
        [
            Chunk(
                repo_id=repo_a_id,
                file_path="src/main.py",
                line_start=10,
                line_end=12,
                content="def alpha():\n    return 1",
                lang="python",
                file_mtime=1.0,
            ),
            Chunk(
                repo_id=repo_a_id,
                file_path="src/helper.ts",
                line_start=20,
                line_end=24,
                content="export const helper = () => 2;",
                lang="typescript",
                file_mtime=1.0,
            ),
            Chunk(
                repo_id=repo_b_id,
                file_path="pkg/main.py",
                line_start=30,
                line_end=34,
                content="def beta():\n    return 2",
                lang="python",
                file_mtime=1.0,
            ),
        ],
        [[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]],
    )
    provider = FakeQueryProvider({"alpha query": [1.0, 0.0]}, dimensions=2)
    searcher = Searcher(storage=storage, provider=provider)

    results = searcher.search(
        "alpha query",
        repo="repo-a",
        langs=["python"],
        path_glob="src/*.py",
        threshold=0.9,
    )

    assert provider.calls == [["alpha query"]]
    assert len(results) == 1
    assert results[0].repo == "repo-a"
    assert results[0].path == "src/main.py"
    assert results[0].line == 10
    assert results[0].lang == "python"


def test_searcher_raises_for_dimension_mismatch_before_query_embed(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo_id = storage.add_repo("repo", str(repo_path))
    storage.insert_chunks(
        [
            Chunk(
                repo_id=repo_id,
                file_path="main.py",
                line_start=1,
                line_end=2,
                content="print('ok')",
                lang="python",
                file_mtime=1.0,
            )
        ],
        [[1.0, 0.0]],
    )
    provider = FakeQueryProvider({"query": [1.0, 0.0, 0.0]}, dimensions=3)
    searcher = Searcher(storage=storage, provider=provider)

    with pytest.raises(DimensionMismatchError, match="Index dimensions 2 do not match query dimensions 3"):
        searcher.search("query")

    assert provider.calls == []


def test_searcher_validates_query_and_repo_filters(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")
    provider = FakeQueryProvider({"query": [1.0, 0.0]}, dimensions=None)
    searcher = Searcher(storage=storage, provider=provider)

    with pytest.raises(CodeSearchError, match="Query cannot be empty"):
        searcher.search("   ")

    with pytest.raises(StorageError, match="Repository not found: missing"):
        searcher.search("query", repo="missing")
