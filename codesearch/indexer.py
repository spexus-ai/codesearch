from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable
import warnings

from codesearch.chunker import Chunker
from codesearch.errors import ChunkerError, DimensionMismatchError, StorageError
from codesearch.embedding.base import EmbeddingProvider
from codesearch.lsh import SimHashLSH
from codesearch.scanner import FileScanner
from codesearch.storage import Chunk, Storage

EMBED_BATCH_SIZE = 32


@dataclass(slots=True)
class IndexResult:
    files_indexed: int
    chunks_created: int
    duration_seconds: float
    scan_seconds: float
    chunk_seconds: float
    embed_seconds: float
    lsh_seconds: float
    store_seconds: float


class Indexer:
    def __init__(
        self,
        storage: Storage,
        provider: EmbeddingProvider,
        scanner: FileScanner,
        chunker: Chunker,
    ):
        self.storage = storage
        self.provider = provider
        self.scanner = scanner
        self.chunker = chunker
        self.lsh, self._lsh_matrix_reset = self._initialize_lsh()

    def _initialize_lsh(self) -> tuple[SimHashLSH, bool]:
        dimensions = self.provider.dimensions()
        stored_params = self.storage.load_lsh_params()
        if stored_params is None:
            lsh = SimHashLSH(dim=dimensions)
            self.storage.save_lsh_params(num_bands=lsh.num_bands, band_width=lsh.band_width)
        else:
            num_bands, band_width = stored_params
            lsh = SimHashLSH(num_bands=num_bands, band_width=band_width, dim=dimensions)

        matrix_blob = self.storage.load_hyperplane_matrix()
        if matrix_blob is None:
            self.storage.save_hyperplane_matrix(lsh.serialize_matrix(lsh._matrix))
            return lsh, False

        matrix = lsh.deserialize_matrix(matrix_blob)
        expected_shape = (lsh.num_bands * lsh.band_width, dimensions)
        if matrix.shape != expected_shape:
            lsh._matrix = lsh.generate_hyperplane_matrix()
            return lsh, True

        lsh._matrix = matrix
        return lsh, False

    def _prepare_lsh_for_indexing(self, *, full: bool) -> None:
        stored_dimensions = self.storage.get_meta("embedding_dimensions")
        dimension_mismatch = stored_dimensions is not None and int(stored_dimensions) != self.lsh.dim
        if dimension_mismatch and not full:
            raise DimensionMismatchError(
                f"Index dimensions {stored_dimensions} do not match current embeddings {self.lsh.dim}. "
                "Run index --full."
            )
        if not full:
            return
        if dimension_mismatch or self._lsh_matrix_reset:
            self.storage.clear_chunk_bands()
        if self._lsh_matrix_reset:
            self.storage.save_hyperplane_matrix(self.lsh.serialize_matrix(self.lsh._matrix))
            self._lsh_matrix_reset = False

    def index_repo(
        self,
        repo_id: int,
        repo_path: Path,
        full: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IndexResult:
        repo_path = Path(repo_path).expanduser().resolve()
        if not repo_path.is_dir():
            raise StorageError(f"Repository directory not found: {repo_path}")
        self._prepare_lsh_for_indexing(full=full)

        started_at = perf_counter()

        scan_started_at = perf_counter()
        scanned_files = self.scanner.scan(repo_path)
        scan_seconds = perf_counter() - scan_started_at

        indexed_files = set(self.storage.list_repo_file_paths(repo_id))
        removed_files = sorted(indexed_files - set(scanned_files))
        if full:
            changed_files = sorted(scanned_files)
            delete_targets = sorted(indexed_files)
        else:
            changed_files = self.storage.get_changed_files(repo_id, scanned_files)
            delete_targets = sorted(set(changed_files) | set(removed_files))

        chunk_band_delete_targets = delete_targets
        if full and self._lsh_matrix_reset:
            chunk_band_delete_targets = sorted(indexed_files)

        if chunk_band_delete_targets:
            self.storage.delete_chunk_bands_for_files(repo_id, chunk_band_delete_targets)
        if delete_targets:
            self.storage.delete_chunks_for_files(repo_id, delete_targets)

        chunk_started_at = perf_counter()
        chunks: list[Chunk] = []
        files_indexed = 0
        files_total = len(changed_files)
        files_done = 0
        for relative_path in changed_files:
            file_path = repo_path / relative_path
            if not file_path.exists():
                files_done += 1
                if progress_callback is not None:
                    progress_callback(files_done, files_total)
                continue
            try:
                file_chunks = self.chunker.chunk_file(file_path, repo_path)
            except ChunkerError as exc:
                warnings.warn(f"Skipping {relative_path}: {exc}", RuntimeWarning, stacklevel=2)
                files_done += 1
                if progress_callback is not None:
                    progress_callback(files_done, files_total)
                continue
            for chunk in file_chunks:
                chunks.append(
                    Chunk(
                        repo_id=repo_id,
                        file_path=chunk.file_path,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        content=chunk.content,
                        lang=chunk.lang,
                        file_mtime=scanned_files[relative_path],
                    )
                )
            files_indexed += 1
            files_done += 1
            if progress_callback is not None:
                progress_callback(files_done, files_total)
        chunk_seconds = perf_counter() - chunk_started_at

        embed_started_at = perf_counter()
        embeddings: list[list[float]] = []
        total_chunks = len(chunks)
        for start in range(0, total_chunks, EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            batch_embeddings = self.provider.embed([chunk.content for chunk in batch])
            if len(batch_embeddings) != len(batch):
                raise StorageError("Embedding provider returned an unexpected number of vectors")
            embeddings.extend(batch_embeddings)
        embed_seconds = perf_counter() - embed_started_at

        lsh_started_at = perf_counter()
        band_hashes = self.lsh.compute_band_hashes(embeddings)
        lsh_seconds = perf_counter() - lsh_started_at

        store_started_at = perf_counter()
        if chunks:
            chunk_ids = self.storage.insert_chunks(chunks, embeddings)
            for chunk_id, chunk_bands in zip(chunk_ids, band_hashes, strict=True):
                self.storage.insert_chunk_bands(chunk_id, chunk_bands)
        self.storage.refresh_repo(repo_id)
        store_seconds = perf_counter() - store_started_at

        return IndexResult(
            files_indexed=files_indexed,
            chunks_created=len(chunks),
            duration_seconds=perf_counter() - started_at,
            scan_seconds=scan_seconds,
            chunk_seconds=chunk_seconds,
            embed_seconds=embed_seconds,
            lsh_seconds=lsh_seconds,
            store_seconds=store_seconds,
        )
