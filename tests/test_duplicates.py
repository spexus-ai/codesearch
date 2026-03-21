from __future__ import annotations

from pathlib import Path

from codesearch.duplicates import DuplicateFinder
from codesearch.lsh import SimHashLSH
from codesearch.storage import Chunk, Storage


def test_duplicate_finder_returns_sorted_pairs_with_metadata_and_limit(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(tmp_path / "repo"))
    chunk_ids = storage.insert_chunks(
        [
            Chunk(repo_id=repo_id, file_path="src/a.py", line_start=1, line_end=2, content="alpha", lang="python", file_mtime=1.0),
            Chunk(repo_id=repo_id, file_path="src/b.py", line_start=3, line_end=4, content="beta", lang="python", file_mtime=1.0),
            Chunk(repo_id=repo_id, file_path="src/a.py", line_start=10, line_end=11, content="gamma", lang="python", file_mtime=1.0),
        ],
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
    )
    for chunk_id in chunk_ids:
        storage.insert_chunk_bands(chunk_id, [(0, b"same-band")])

    finder = DuplicateFinder(storage=storage, lsh=SimHashLSH(dim=2))

    pairs = finder.find_duplicates(threshold=0.99, limit=2)

    assert len(pairs) == 2
    assert [pair.similarity for pair in pairs] == [1.0, 1.0]
    assert pairs[0].chunk_a.path == "src/a.py"
    assert pairs[0].chunk_a.snippet == "alpha"
    assert pairs[0].chunk_b.path == "src/b.py"
    assert pairs[0].chunk_b.lang == "python"


def test_duplicate_finder_respects_cross_file_only_and_empty_candidates(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(tmp_path / "repo"))
    chunk_ids = storage.insert_chunks(
        [
            Chunk(repo_id=repo_id, file_path="src/a.py", line_start=1, line_end=2, content="alpha", lang="python", file_mtime=1.0),
            Chunk(repo_id=repo_id, file_path="src/b.py", line_start=3, line_end=4, content="beta", lang="python", file_mtime=1.0),
            Chunk(repo_id=repo_id, file_path="src/a.py", line_start=10, line_end=11, content="gamma", lang="python", file_mtime=1.0),
        ],
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
    )
    for chunk_id in chunk_ids:
        storage.insert_chunk_bands(chunk_id, [(0, b"same-band")])

    finder = DuplicateFinder(storage=storage, lsh=SimHashLSH(dim=2))

    cross_file_pairs = finder.find_duplicates(cross_file_only=True, threshold=0.99, limit=10)
    filtered_pairs = finder.find_duplicates(path_globs=["missing/*"], threshold=0.99, limit=10)

    assert [(pair.chunk_a.path, pair.chunk_b.path) for pair in cross_file_pairs] == [
        ("src/a.py", "src/b.py"),
        ("src/b.py", "src/a.py"),
    ]
    assert filtered_pairs == []
