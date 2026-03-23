from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from codesearch.chunker import Chunker
from codesearch.errors import DimensionMismatchError
from codesearch.indexer import Indexer
from codesearch.scanner import FileScanner
from codesearch.storage import Storage


class FakeEmbeddingProvider:
    def __init__(self, dimensions: int = 3):
        self._dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        embeddings: list[list[float]] = []
        for index, text in enumerate(texts):
            base = [float(len(text)), float(index), float(self._dimensions)]
            if self._dimensions <= len(base):
                embeddings.append(base[: self._dimensions])
                continue
            embeddings.append(base + [0.0] * (self._dimensions - len(base)))
        return embeddings

    def dimensions(self) -> int:
        return self._dimensions


class InterruptingEmbeddingProvider(FakeEmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise KeyboardInterrupt


class TrackingStorage(Storage):
    def __init__(self, db_path: Path):
        super().__init__(db_path)
        self.events: list[tuple[str, int, tuple[str, ...]]] = []
        self.inserted_chunk_bands: list[tuple[int, list[tuple[int, bytes]]]] = []

    def delete_chunk_bands_for_files(self, repo_id: int, file_paths: list[str]) -> None:
        self.events.append(("delete_chunk_bands_for_files", repo_id, tuple(file_paths)))
        super().delete_chunk_bands_for_files(repo_id, file_paths)

    def clear_chunk_bands(self) -> None:
        self.events.append(("clear_chunk_bands", 0, ()))
        super().clear_chunk_bands()

    def delete_chunks_for_files(self, repo_id: int, file_paths: list[str]) -> int:
        self.events.append(("delete_chunks_for_files", repo_id, tuple(file_paths)))
        return super().delete_chunks_for_files(repo_id, file_paths)

    def insert_chunk_bands(self, chunk_id: int, bands: list[tuple[int, bytes]]) -> None:
        self.inserted_chunk_bands.append((chunk_id, list(bands)))
        super().insert_chunk_bands(chunk_id, bands)


def _row_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


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
    assert result.lsh_seconds >= 0.0
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


def test_indexer_skips_binary_files(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    binary_file = repo_path / "binary.py"
    binary_file.write_bytes(b"\xff\xfe\x00\x00")

    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(repo_path))
    provider = FakeEmbeddingProvider()
    indexer = Indexer(storage=storage, provider=provider, scanner=FileScanner(), chunker=Chunker())

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


def test_indexer_initializes_lsh_metadata_and_persists_chunk_bands(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("key = 'alpha'\n", encoding="utf-8")

    storage = TrackingStorage(db_path)
    repo_id = storage.add_repo("repo", str(repo_path))
    provider = FakeEmbeddingProvider()

    assert storage.load_hyperplane_matrix() is None
    assert storage.load_lsh_params() is None

    indexer = Indexer(storage=storage, provider=provider, scanner=FileScanner(), chunker=Chunker())
    first_matrix_blob = storage.load_hyperplane_matrix()

    assert first_matrix_blob is not None
    assert storage.load_lsh_params() == (indexer.lsh.num_bands, indexer.lsh.band_width)

    result = indexer.index_repo(repo_id, repo_path)

    reloaded = Indexer(storage=storage, provider=provider, scanner=FileScanner(), chunker=Chunker())

    assert result.chunks_created == 1
    assert result.lsh_seconds >= 0.0
    assert _row_count(db_path, "chunk_bands") == indexer.lsh.num_bands
    assert len(storage.inserted_chunk_bands) == 1
    assert len(storage.inserted_chunk_bands[0][1]) == indexer.lsh.num_bands
    assert storage.load_hyperplane_matrix() == first_matrix_blob
    assert reloaded.lsh.dim == provider.dimensions()


def test_indexer_deletes_chunk_bands_before_deleting_chunks_on_incremental_reindex(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "index.db"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    alpha = repo_path / "alpha.toml"
    beta = repo_path / "beta.toml"
    alpha.write_text("key = 'alpha'\n", encoding="utf-8")
    beta.write_text("key = 'beta'\n", encoding="utf-8")

    storage = TrackingStorage(db_path)
    repo_id = storage.add_repo("repo", str(repo_path))
    provider = FakeEmbeddingProvider()
    indexer = Indexer(storage=storage, provider=provider, scanner=FileScanner(), chunker=Chunker())

    indexer.index_repo(repo_id, repo_path)
    storage.events.clear()

    alpha.write_text("key = 'alpha-updated'\n", encoding="utf-8")
    os.utime(alpha, (200.0, 200.0))
    beta.unlink()

    indexer.index_repo(repo_id, repo_path)

    assert storage.events[:2] == [
        ("delete_chunk_bands_for_files", repo_id, ("alpha.toml", "beta.toml")),
        ("delete_chunks_for_files", repo_id, ("alpha.toml", "beta.toml")),
    ]
    assert _row_count(db_path, "chunk_bands") == indexer.lsh.num_bands


def test_indexer_full_reindex_regenerates_lsh_matrix_for_dimension_change(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("key = 'alpha'\n", encoding="utf-8")

    storage = TrackingStorage(db_path)
    repo_id = storage.add_repo("repo", str(repo_path))
    initial_provider = FakeEmbeddingProvider(dimensions=3)
    initial_indexer = Indexer(
        storage=storage,
        provider=initial_provider,
        scanner=FileScanner(),
        chunker=Chunker(),
    )

    initial_indexer.index_repo(repo_id, repo_path)
    original_matrix_blob = storage.load_hyperplane_matrix()
    storage.events.clear()

    updated_provider = FakeEmbeddingProvider(dimensions=4)
    updated_indexer = Indexer(
        storage=storage,
        provider=updated_provider,
        scanner=FileScanner(),
        chunker=Chunker(),
    )
    with pytest.raises(DimensionMismatchError, match="Run index --full"):
        updated_indexer.index_repo(repo_id, repo_path)

    updated_matrix_blob = storage.load_hyperplane_matrix()

    result = updated_indexer.index_repo(repo_id, repo_path, full=True)
    final_matrix_blob = storage.load_hyperplane_matrix()

    assert original_matrix_blob is not None
    assert updated_matrix_blob == original_matrix_blob
    assert final_matrix_blob is not None
    assert final_matrix_blob != original_matrix_blob
    assert updated_indexer.lsh._matrix.shape == (
        updated_indexer.lsh.num_bands * updated_indexer.lsh.band_width,
        4,
    )
    assert storage.events[:3] == [
        ("clear_chunk_bands", 0, ()),
        ("delete_chunk_bands_for_files", repo_id, ("alpha.toml",)),
        ("delete_chunks_for_files", repo_id, ("alpha.toml",)),
    ]
    assert storage.get_meta("embedding_dimensions") == "4"
    assert result.lsh_seconds >= 0.0
    assert _row_count(db_path, "chunk_bands") == updated_indexer.lsh.num_bands
