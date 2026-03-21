from __future__ import annotations

from dataclasses import dataclass

from codesearch.lsh import SimHashLSH
from codesearch.storage import ChunkRecord, Storage


@dataclass(slots=True)
class ChunkInfo:
    repo: str
    path: str
    line_start: int
    line_end: int
    snippet: str
    lang: str


@dataclass(slots=True)
class DuplicatePair:
    chunk_a: ChunkInfo
    chunk_b: ChunkInfo
    similarity: float


class DuplicateFinder:
    def __init__(self, storage: Storage, lsh: SimHashLSH):
        self.storage = storage
        self.lsh = lsh

    def find_duplicates(
        self,
        repo_ids: list[int] | None = None,
        path_globs: list[str] | None = None,
        cross_file_only: bool = False,
        threshold: float = 0.95,
        limit: int = 50,
    ) -> list[DuplicatePair]:
        if limit <= 0:
            return []

        candidates = self.storage.find_duplicate_candidates(repo_ids=repo_ids, path_globs=path_globs)
        if not candidates:
            return []

        chunk_ids = sorted({chunk_id for pair in candidates for chunk_id in pair})
        embeddings = self.storage.load_embeddings_for_chunks(chunk_ids)
        verified = self.lsh.verify_pairs(candidates, embeddings, threshold=threshold)
        if not verified:
            return []

        records = self.storage.load_chunk_records(chunk_ids)
        pairs: list[DuplicatePair] = []
        seen_pairs: set[tuple[int, int]] = set()
        for chunk_id_a, chunk_id_b, similarity in verified:
            if (chunk_id_a, chunk_id_b) in seen_pairs:
                continue
            record_a = records.get(chunk_id_a)
            record_b = records.get(chunk_id_b)
            if record_a is None or record_b is None:
                continue
            if cross_file_only and self._same_file(record_a, record_b):
                continue
            pairs.append(
                DuplicatePair(
                    chunk_a=self._to_chunk_info(record_a),
                    chunk_b=self._to_chunk_info(record_b),
                    similarity=similarity,
                )
            )
            seen_pairs.add((chunk_id_a, chunk_id_b))

        pairs.sort(key=lambda pair: pair.similarity, reverse=True)
        return pairs[:limit]

    def _to_chunk_info(self, record: ChunkRecord) -> ChunkInfo:
        return ChunkInfo(
            repo=record.repo,
            path=record.path,
            line_start=record.line_start,
            line_end=record.line_end,
            snippet=record.content,
            lang=record.lang,
        )

    def _same_file(self, left: ChunkRecord, right: ChunkRecord) -> bool:
        return left.repo == right.repo and left.path == right.path
