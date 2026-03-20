from __future__ import annotations

import os
from pathlib import Path

import pytest

from codesearch.chunker import Chunker
from codesearch.indexer import Indexer
from codesearch.scanner import FileScanner
from codesearch.storage import Storage


class FakeEmbeddingProvider:
    def __init__(self, dimensions: int = 3):
        self._dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), float(index), float(self._dimensions)] for index, text in enumerate(texts)]

    def dimensions(self) -> int:
        return self._dimensions


class InterruptingEmbeddingProvider(FakeEmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise KeyboardInterrupt


def test_indexer_indexes_changed_files_in_batches_and_updates_storage(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    for index in range(33):
        (repo_path / f"file_{index}.toml").write_text(f"key = 'value-{index}'\n", encoding="utf-8")

    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(repo_path))
    provider = FakeEmbeddingProvider()
    progress_events: list[tuple[int, int]] = []
    indexer = Indexer(storage=storage, provider=provider, scanner=FileScanner(), chunker=Chunker())

    result = indexer.index_repo(
        repo_id,
        repo_path,
        progress_callback=lambda processed, total: progress_events.append((processed, total)),
    )
    repos = storage.list_repos()

    assert result.files_indexed == 33
    assert result.chunks_created == 33
    assert result.scan_seconds >= 0.0
    assert result.chunk_seconds >= 0.0
    assert result.embed_seconds >= 0.0
    assert result.store_seconds >= 0.0
    assert storage.get_meta("embedding_dimensions") == "3"
    assert [len(call) for call in provider.calls] == [32, 1]
    assert progress_events == [(index, 33) for index in range(1, 34)]
    assert repos[0].file_count == 33
    assert repos[0].chunk_count == 33
    assert repos[0].indexed_at is not None


def test_indexer_reindexes_only_changed_files_and_removes_deleted_files(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    alpha = repo_path / "alpha.toml"
    beta = repo_path / "beta.toml"
    alpha.write_text("key = 'alpha'\n", encoding="utf-8")
    beta.write_text("key = 'beta'\n", encoding="utf-8")

    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(repo_path))
    provider = FakeEmbeddingProvider()
    indexer = Indexer(storage=storage, provider=provider, scanner=FileScanner(), chunker=Chunker())

    first = indexer.index_repo(repo_id, repo_path)
    alpha.write_text("key = 'alpha-updated'\n", encoding="utf-8")
    os.utime(alpha, (200.0, 200.0))
    beta.unlink()
    provider.calls.clear()

    second = indexer.index_repo(repo_id, repo_path)
    repos = storage.list_repos()
    results = storage.search([20.0, 0.0, 3.0], limit=5)

    assert first.files_indexed == 2
    assert first.chunks_created == 2
    assert second.files_indexed == 1
    assert second.chunks_created == 1
    assert [len(call) for call in provider.calls] == [1]
    assert repos[0].file_count == 1
    assert repos[0].chunk_count == 1
    assert storage.list_repo_file_paths(repo_id) == ["alpha.toml"]
    assert len(results) == 1
    assert results[0].path == "alpha.toml"


def test_indexer_full_reindexes_all_files(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    alpha = repo_path / "alpha.toml"
    beta = repo_path / "beta.toml"
    alpha.write_text("key = 'alpha'\n", encoding="utf-8")
    beta.write_text("key = 'beta'\n", encoding="utf-8")

    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(repo_path))
    provider = FakeEmbeddingProvider()
    indexer = Indexer(storage=storage, provider=provider, scanner=FileScanner(), chunker=Chunker())

    first = indexer.index_repo(repo_id, repo_path)
    provider.calls.clear()
    beta.write_text("key = 'beta-updated'\n", encoding="utf-8")
    os.utime(beta, (200.0, 200.0))

    second = indexer.index_repo(repo_id, repo_path, full=True)

    assert first.files_indexed == 2
    assert second.files_indexed == 2
    assert second.chunks_created == 2
    assert [len(call) for call in provider.calls] == [2]
    assert storage.list_repos()[0].chunk_count == 2


def test_indexer_warns_and_skips_binary_files(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    binary_file = repo_path / "binary.py"
    binary_file.write_bytes(b"\xff\xfe\x00\x00")

    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(repo_path))
    provider = FakeEmbeddingProvider()
    indexer = Indexer(storage=storage, provider=provider, scanner=FileScanner(), chunker=Chunker())

    with pytest.warns(RuntimeWarning, match="Skipping binary.py"):
        result = indexer.index_repo(repo_id, repo_path)

    assert result.files_indexed == 0
    assert result.chunks_created == 0
    assert provider.calls == []
    assert storage.list_repos()[0].chunk_count == 0


def test_indexer_uses_line_fallback_when_ast_chunking_fails(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    source = repo_path / "main.py"
    source.write_text("\n".join(f"value_{idx} = {idx}" for idx in range(1, 101)), encoding="utf-8")

    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(repo_path))
    provider = FakeEmbeddingProvider()
    chunker = Chunker()

    class BrokenParser:
        def parse(self, _: bytes):
            raise RuntimeError("boom")

    chunker._parsers["python"] = BrokenParser()
    indexer = Indexer(storage=storage, provider=provider, scanner=FileScanner(), chunker=chunker)

    result = indexer.index_repo(repo_id, repo_path)
    indexed_paths = storage.list_repo_file_paths(repo_id)

    assert result.files_indexed == 1
    assert result.chunks_created == 2
    assert [len(call) for call in provider.calls] == [2]
    assert indexed_paths == ["main.py"]


def test_indexer_leaves_index_consistent_on_keyboard_interrupt(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")

    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(repo_path))
    indexer = Indexer(
        storage=storage,
        provider=InterruptingEmbeddingProvider(),
        scanner=FileScanner(),
        chunker=Chunker(),
    )

    with pytest.raises(KeyboardInterrupt):
        indexer.index_repo(repo_id, repo_path)

    repo = storage.list_repos()[0]
    assert repo.file_count == 0
    assert repo.chunk_count == 0
    assert storage.list_repo_file_paths(repo_id) == []
