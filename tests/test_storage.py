import sqlite3
from pathlib import Path

import pytest

from codesearch.errors import DimensionMismatchError, StorageError
from codesearch.storage import Chunk, Storage


def _connect_with_vec(db_path: Path) -> sqlite3.Connection:
    import sqlite_vec

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _row_count(db_path: Path, table: str) -> int:
    with _connect_with_vec(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_storage_initializes_schema_on_first_use(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    storage = Storage(db_path)

    repos = storage.list_repos()

    assert repos == []
    assert db_path.exists()
    assert storage.get_meta("schema_version") == "1"
    assert storage.vector_backend == "sqlite-vec"
    assert storage.get_meta("vec_dimensions") == "768"

    with _connect_with_vec(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            ).fetchall()
        }
        vec_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
        ).fetchone()[0]

    assert "meta" in tables
    assert "repos" in tables
    assert "chunks" in tables
    assert "vec_chunks" in tables
    assert "idx_chunks_repo" in tables
    assert "idx_chunks_file" in tables
    assert "idx_chunks_lang" in tables
    assert "USING vec0" in vec_sql
    assert "embedding float[768]" in vec_sql


def test_add_repo_persists_fields_and_returns_id(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    storage = Storage(db_path)

    repo_id = storage.add_repo("repo", str(repo_path))
    repos = storage.list_repos()

    assert repo_id == 1
    assert len(repos) == 1
    assert repos[0].id == repo_id
    assert repos[0].name == "repo"
    assert repos[0].path == str(repo_path.resolve())
    assert repos[0].indexed_at is None
    assert repos[0].file_count == 0
    assert repos[0].chunk_count == 0


def test_add_repo_rejects_duplicate_path(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    storage = Storage(db_path)
    storage.add_repo("repo", str(repo_path))

    with pytest.raises(StorageError, match="Repository already registered: alias"):
        storage.add_repo("alias", str(repo_path))


def test_add_repo_rejects_duplicate_name(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    storage = Storage(db_path)
    storage.add_repo("repo", str(repo_a))

    with pytest.raises(StorageError, match="Repository already registered: repo"):
        storage.add_repo("repo", str(repo_b))


def test_list_repos_returns_multiple_entries_sorted_by_name(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_a = tmp_path / "a-repo"
    repo_b = tmp_path / "b-repo"
    repo_a.mkdir()
    repo_b.mkdir()
    storage = Storage(db_path)
    storage.add_repo("zeta", str(repo_b))
    storage.add_repo("alpha", str(repo_a))

    repos = storage.list_repos()

    assert [repo.name for repo in repos] == ["alpha", "zeta"]


def test_repo_lifecycle_and_cascade_cleanup(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    storage = Storage(db_path)

    repo_id = storage.add_repo("repo-a", str(repo_path))
    inserted_ids = storage.insert_chunks(
        [
            Chunk(
                repo_id=repo_id,
                file_path="src/app.py",
                line_start=1,
                line_end=4,
                content="def main():\n    return 'a'",
                lang="python",
                file_mtime=100.0,
            ),
            Chunk(
                repo_id=repo_id,
                file_path="src/util.py",
                line_start=1,
                line_end=3,
                content="def util():\n    return 'b'",
                lang="python",
                file_mtime=101.0,
            ),
        ],
        [[1.0, 0.0], [0.8, 0.2]],
    )

    repos = storage.list_repos()

    assert len(inserted_ids) == 2
    assert len(repos) == 1
    assert repos[0].name == "repo-a"
    assert repos[0].file_count == 2
    assert repos[0].chunk_count == 2
    assert repos[0].indexed_at is not None
    assert _row_count(db_path, "chunks") == 2
    assert _row_count(db_path, "vec_chunks") == 2

    removed = storage.remove_repo("repo-a")

    assert removed == 2
    assert storage.list_repos() == []
    assert _row_count(db_path, "repos") == 0
    assert _row_count(db_path, "chunks") == 0
    assert _row_count(db_path, "vec_chunks") == 0


def test_remove_repo_by_path(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    storage = Storage(db_path)
    storage.add_repo("repo-a", str(repo_path))

    removed = storage.remove_repo(str(repo_path))

    assert removed == 0
    assert storage.list_repos() == []


def test_remove_repo_not_found_returns_zero(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "index.db")

    removed = storage.remove_repo("missing")

    assert removed == 0


def test_get_changed_files_and_delete_chunks_for_files(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_path = tmp_path / "repo-b"
    repo_path.mkdir()
    storage = Storage(db_path)
    repo_id = storage.add_repo("repo-b", str(repo_path))

    storage.insert_chunks(
        [
            Chunk(
                repo_id=repo_id,
                file_path="a.py",
                line_start=1,
                line_end=2,
                content="print('a')",
                lang="python",
                file_mtime=10.0,
            ),
            Chunk(
                repo_id=repo_id,
                file_path="a.py",
                line_start=3,
                line_end=4,
                content="print('b')",
                lang="python",
                file_mtime=10.0,
            ),
            Chunk(
                repo_id=repo_id,
                file_path="b.py",
                line_start=1,
                line_end=2,
                content="print('c')",
                lang="python",
                file_mtime=20.0,
            ),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )

    changed = storage.get_changed_files(
        repo_id,
        {
            "a.py": 10.0,
            "b.py": 22.0,
            "c.py": 1.0,
        },
    )

    assert changed == ["b.py", "c.py"]

    deleted = storage.delete_chunks_for_files(repo_id, ["a.py"])
    repos = storage.list_repos()

    assert deleted == 2
    assert _row_count(db_path, "chunks") == 1
    assert _row_count(db_path, "vec_chunks") == 1
    assert repos[0].file_count == 1
    assert repos[0].chunk_count == 1


def test_search_supports_metadata_filters_and_threshold(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_a_path = tmp_path / "repo-a"
    repo_b_path = tmp_path / "repo-b"
    repo_a_path.mkdir()
    repo_b_path.mkdir()
    storage = Storage(db_path)
    repo_a = storage.add_repo("repo-a", str(repo_a_path))
    repo_b = storage.add_repo("repo-b", str(repo_b_path))

    storage.insert_chunks(
        [
            Chunk(
                repo_id=repo_a,
                file_path="src/main.py",
                line_start=10,
                line_end=12,
                content="def alpha():\n    return 1",
                lang="python",
                file_mtime=1.0,
            ),
            Chunk(
                repo_id=repo_a,
                file_path="src/other.ts",
                line_start=20,
                line_end=23,
                content="export const beta = () => 2;",
                lang="typescript",
                file_mtime=1.0,
            ),
            Chunk(
                repo_id=repo_b,
                file_path="pkg/main.py",
                line_start=30,
                line_end=33,
                content="def gamma():\n    return 3",
                lang="python",
                file_mtime=1.0,
            ),
        ],
        [[1.0, 0.0], [0.6, 0.4], [0.0, 1.0]],
    )

    baseline = storage.search([1.0, 0.0], limit=2, threshold=0.7)
    repo_filtered = storage.search([1.0, 0.0], repo_id=repo_a, langs=["python"], threshold=0.7)
    glob_filtered = storage.search([1.0, 0.0], path_glob="src/*.py", threshold=0.7)

    assert [item.path for item in baseline] == ["src/main.py", "src/other.ts"]
    assert len(repo_filtered) == 1
    assert repo_filtered[0].repo == "repo-a"
    assert repo_filtered[0].path == "src/main.py"
    assert len(glob_filtered) == 1
    assert glob_filtered[0].path == "src/main.py"


def test_storage_guards_embedding_dimensions(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    repo_path = tmp_path / "repo-c"
    repo_path.mkdir()
    storage = Storage(db_path)
    repo_id = storage.add_repo("repo-c", str(repo_path))

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

    with pytest.raises(DimensionMismatchError, match="Index dimensions 2 do not match current embeddings 3"):
        storage.insert_chunks(
            [
                Chunk(
                    repo_id=repo_id,
                    file_path="next.py",
                    line_start=1,
                    line_end=2,
                    content="print('fail')",
                    lang="python",
                    file_mtime=2.0,
                )
            ],
            [[1.0, 0.0, 0.0]],
        )

    with pytest.raises(DimensionMismatchError, match="Index dimensions 2 do not match query dimensions 3"):
        storage.search([1.0, 0.0, 0.0])


def test_storage_raises_when_sqlite_vec_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = Storage(tmp_path / "index.db")

    def _raise_missing(name: str):
        if name == "sqlite_vec":
            raise ModuleNotFoundError(name)
        return __import__(name)

    monkeypatch.setattr("codesearch.storage.importlib.import_module", _raise_missing)

    with pytest.raises(StorageError, match="sqlite-vec extension not found. Install: pip install sqlite-vec"):
        storage.list_repos()


def test_storage_raises_for_corrupted_database(tmp_path: Path) -> None:
    db_path = tmp_path / "broken.db"
    db_path.write_text("not a sqlite database", encoding="utf-8")
    storage = Storage(db_path)

    with pytest.raises(StorageError, match="Database corrupted. Delete ~/.codesearch/index.db and re-index."):
        storage.list_repos()


def test_storage_raises_for_locked_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = Storage(tmp_path / "index.db")
    real_connect = sqlite3.connect

    def _locked_connect(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("codesearch.storage.sqlite3.connect", _locked_connect)

    with pytest.raises(StorageError, match="Database is locked by another process"):
        storage.list_repos()

    monkeypatch.setattr("codesearch.storage.sqlite3.connect", real_connect)
