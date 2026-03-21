from __future__ import annotations

import sqlite3
from pathlib import Path

from codesearch.storage import Chunk, Storage


def _row_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _seed_chunks(storage: Storage, repo_id: int, base_path: str = "src") -> list[int]:
    return storage.insert_chunks(
        [
            Chunk(repo_id=repo_id, file_path=f"{base_path}/a.py", line_start=1, line_end=2, content="alpha", lang="python", file_mtime=1.0),
            Chunk(repo_id=repo_id, file_path=f"{base_path}/b.py", line_start=3, line_end=4, content="beta", lang="python", file_mtime=1.0),
            Chunk(repo_id=repo_id, file_path=f"{base_path}/c.py", line_start=5, line_end=6, content="gamma", lang="python", file_mtime=1.0),
        ],
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ],
    )


def test_create_chunk_bands_table(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    storage = Storage(db_path)

    storage.list_repos()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'chunk_bands'"
        ).fetchone()

    assert row == ("chunk_bands",)


def test_insert_and_query_chunk_bands(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(tmp_path / "repo"))
    chunk_ids = _seed_chunks(storage, repo_id)

    storage.insert_chunk_bands(chunk_ids[0], [(0, b"band-a")])
    storage.insert_chunk_bands(chunk_ids[1], [(0, b"band-a")])
    storage.insert_chunk_bands(chunk_ids[2], [(0, b"band-b")])

    assert storage.find_duplicate_candidates(repo_ids=None, path_globs=None) == [
        (chunk_ids[0], chunk_ids[1])
    ]


def test_find_candidates_with_repo_filter(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")
    repo_a = storage.add_repo("repo-a", str(tmp_path / "repo-a"))
    repo_b = storage.add_repo("repo-b", str(tmp_path / "repo-b"))
    chunk_ids_a = _seed_chunks(storage, repo_a)
    chunk_ids_b = _seed_chunks(storage, repo_b)

    storage.insert_chunk_bands(chunk_ids_a[0], [(0, b"band-a")])
    storage.insert_chunk_bands(chunk_ids_a[1], [(0, b"band-a")])
    storage.insert_chunk_bands(chunk_ids_b[0], [(0, b"band-a")])
    storage.insert_chunk_bands(chunk_ids_b[1], [(0, b"band-a")])

    assert storage.find_duplicate_candidates(repo_ids=[repo_a], path_globs=None) == [
        (chunk_ids_a[0], chunk_ids_a[1])
    ]


def test_find_candidates_with_path_filter(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(tmp_path / "repo"))
    src_chunk_ids = _seed_chunks(storage, repo_id, base_path="src")
    test_chunk_ids = _seed_chunks(storage, repo_id, base_path="tests")

    storage.insert_chunk_bands(src_chunk_ids[0], [(0, b"band-a")])
    storage.insert_chunk_bands(src_chunk_ids[1], [(0, b"band-a")])
    storage.insert_chunk_bands(test_chunk_ids[0], [(0, b"band-a")])
    storage.insert_chunk_bands(test_chunk_ids[1], [(0, b"band-a")])

    assert storage.find_duplicate_candidates(repo_ids=None, path_globs=["src/*.py"]) == [
        (src_chunk_ids[0], src_chunk_ids[1])
    ]


def test_delete_chunk_bands_for_files(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    storage = Storage(db_path)
    repo_id = storage.add_repo("repo", str(tmp_path / "repo"))
    chunk_ids = _seed_chunks(storage, repo_id)

    storage.insert_chunk_bands(chunk_ids[0], [(0, b"band-a")])
    storage.insert_chunk_bands(chunk_ids[1], [(0, b"band-a")])
    storage.insert_chunk_bands(chunk_ids[2], [(0, b"band-b")])

    storage.delete_chunk_bands_for_files(repo_id, ["src/a.py"])

    assert _row_count(db_path, "chunk_bands") == 2
    assert storage.find_duplicate_candidates(repo_ids=None, path_globs=None) == []


def test_hyperplane_matrix_persistence(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")

    storage.save_hyperplane_matrix(b"matrix")

    assert storage.load_hyperplane_matrix() == b"matrix"


def test_load_embeddings_for_chunks(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")
    repo_id = storage.add_repo("repo", str(tmp_path / "repo"))
    chunk_ids = _seed_chunks(storage, repo_id)

    assert storage.load_embeddings_for_chunks([chunk_ids[0], chunk_ids[2]]) == {
        chunk_ids[0]: [1.0, 0.0],
        chunk_ids[2]: [0.8, 0.2],
    }
