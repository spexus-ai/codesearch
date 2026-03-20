from __future__ import annotations

import importlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from codesearch.errors import DimensionMismatchError, StorageError

SCHEMA_VERSION = "1"
DEFAULT_VECTOR_DIMENSIONS = 768


@dataclass(slots=True)
class Repo:
    id: int
    name: str
    path: str
    indexed_at: str | None
    file_count: int
    chunk_count: int


@dataclass(slots=True)
class Chunk:
    repo_id: int
    file_path: str
    line_start: int
    line_end: int
    content: str
    lang: str
    file_mtime: float
    id: int | None = None


@dataclass(slots=True)
class SearchResult:
    repo: str
    path: str
    line: int
    score: float
    snippet: str
    lang: str


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).expanduser()
        self._initialized = False

    @property
    def vector_backend(self) -> str:
        self._ensure_initialized()
        return "sqlite-vec"

    def add_repo(self, name: str, path: str) -> int:
        self._ensure_initialized()
        repo_path = str(Path(path).expanduser().resolve())
        sql = """
            INSERT INTO repos(name, path, indexed_at, file_count, chunk_count)
            VALUES (?, ?, NULL, 0, 0)
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(sql, (name, repo_path))
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"Repository already registered: {name}") from exc
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to add repository: {exc}") from exc

    def list_repos(self) -> list[Repo]:
        self._ensure_initialized()
        sql = """
            SELECT id, name, path, indexed_at, file_count, chunk_count
            FROM repos
            ORDER BY name ASC
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(sql).fetchall()
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to list repositories: {exc}") from exc
        return [self._row_to_repo(row) for row in rows]

    def remove_repo(self, name_or_path: str) -> int:
        self._ensure_initialized()
        repo_path = str(Path(name_or_path).expanduser().resolve())
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id FROM repos WHERE name = ? OR path = ?",
                    (name_or_path, repo_path),
                ).fetchone()
                if row is None:
                    return 0
                repo_id = int(row["id"])
                chunk_row = conn.execute(
                    "SELECT COUNT(*) AS chunk_count FROM chunks WHERE repo_id = ?",
                    (repo_id,),
                ).fetchone()
                deleted_chunks = int(chunk_row["chunk_count"]) if chunk_row is not None else 0
                conn.execute(
                    """
                    DELETE FROM vec_chunks
                    WHERE chunk_id IN (
                        SELECT id FROM chunks WHERE repo_id = ?
                    )
                    """,
                    (repo_id,),
                )
                conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
                conn.execute("DELETE FROM repos WHERE id = ?", (repo_id,))
                return deleted_chunks
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to remove repository: {exc}") from exc

    def get_changed_files(self, repo_id: int, files: dict[str, float]) -> list[str]:
        self._ensure_initialized()
        if not files:
            return []
        sql = """
            SELECT file_path, MAX(file_mtime) AS file_mtime
            FROM chunks
            WHERE repo_id = ?
            GROUP BY file_path
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, (repo_id,)).fetchall()
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to read indexed file state: {exc}") from exc

        indexed = {str(row["file_path"]): float(row["file_mtime"]) for row in rows}
        changed: list[str] = []
        for file_path, mtime in files.items():
            known_mtime = indexed.get(file_path)
            if known_mtime is None or abs(known_mtime - mtime) > 1e-9:
                changed.append(file_path)
        return sorted(changed)

    def list_repo_file_paths(self, repo_id: int) -> list[str]:
        self._ensure_initialized()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT file_path
                    FROM chunks
                    WHERE repo_id = ?
                    ORDER BY file_path ASC
                    """,
                    (repo_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to list indexed files: {exc}") from exc
        return [str(row["file_path"]) for row in rows]

    def delete_chunks_for_files(self, repo_id: int, file_paths: Iterable[str]) -> int:
        self._ensure_initialized()
        normalized = sorted({path for path in file_paths})
        if not normalized:
            return 0
        placeholders = ", ".join("?" for _ in normalized)
        params = [repo_id, *normalized]
        try:
            with self._connect() as conn:
                conn.execute(
                    f"""
                    DELETE FROM vec_chunks
                    WHERE chunk_id IN (
                        SELECT id FROM chunks
                        WHERE repo_id = ? AND file_path IN ({placeholders})
                    )
                    """,
                    params,
                )
                cursor = conn.execute(
                    f"DELETE FROM chunks WHERE repo_id = ? AND file_path IN ({placeholders})",
                    params,
                )
                self._refresh_repo_stats(conn, {repo_id})
                return int(cursor.rowcount)
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to delete chunks: {exc}") from exc

    def insert_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> list[int]:
        self._ensure_initialized()
        if len(chunks) != len(embeddings):
            raise StorageError("Chunk count and embedding count must match")
        if not chunks:
            return []

        dimensions = self._validate_dimensions(embeddings)
        repo_ids = {chunk.repo_id for chunk in chunks}

        try:
            with self._connect() as conn:
                self._ensure_embedding_dimensions(conn, dimensions)
                serialize_embedding = self._sqlite_vec_serialize_float32()
                inserted_ids: list[int] = []
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    cursor = conn.execute(
                        """
                        INSERT INTO chunks(repo_id, file_path, line_start, line_end, content, lang, file_mtime)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.repo_id,
                            chunk.file_path,
                            chunk.line_start,
                            chunk.line_end,
                            chunk.content,
                            chunk.lang,
                            chunk.file_mtime,
                        ),
                    )
                    chunk_id = int(cursor.lastrowid)
                    conn.execute(
                        """
                        INSERT INTO vec_chunks(chunk_id, embedding, repo_id, lang, file_path)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            serialize_embedding([float(value) for value in embedding]),
                            chunk.repo_id,
                            chunk.lang,
                            chunk.file_path,
                        ),
                    )
                    inserted_ids.append(chunk_id)
                self._refresh_repo_stats(conn, repo_ids)
                return inserted_ids
        except sqlite3.IntegrityError as exc:
            raise self._storage_error(exc, f"Failed to insert chunks: {exc}") from exc
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to insert chunks: {exc}") from exc

    def search(
        self,
        embedding: Sequence[float],
        limit: int = 10,
        threshold: float = 0.0,
        repo_id: int | None = None,
        langs: Sequence[str] | None = None,
        path_glob: str | None = None,
    ) -> list[SearchResult]:
        self._ensure_initialized()
        if limit <= 0:
            return []

        query_vector = [float(value) for value in embedding]
        self._assert_query_dimensions(query_vector)
        if not self._has_indexed_chunks():
            return []

        k = self._candidate_count(limit, repo_id, langs, path_glob)
        sql = [
            """
            SELECT *
            FROM (
                SELECT
                    repos.name AS repo_name,
                    chunks.repo_id AS repo_id,
                    chunks.file_path AS file_path,
                    chunks.line_start AS line_start,
                    chunks.content AS content,
                    chunks.lang AS lang,
                    vec_chunks.distance AS distance
                FROM vec_chunks
                INNER JOIN chunks ON chunks.id = vec_chunks.chunk_id
                INNER JOIN repos ON repos.id = chunks.repo_id
                WHERE vec_chunks.embedding MATCH ? AND k = ?
                ORDER BY vec_chunks.distance ASC
            )
            WHERE 1 = 1
            """
        ]
        params: list[Any] = [self._serialize_query_embedding(query_vector), k]

        if repo_id is not None:
            sql.append("AND repo_id = ?")
            params.append(repo_id)

        if langs:
            placeholders = ", ".join("?" for _ in langs)
            sql.append(f"AND lang IN ({placeholders})")
            params.extend(langs)

        if path_glob:
            sql.append("AND file_path GLOB ?")
            params.append(path_glob)

        sql.append("ORDER BY distance ASC")

        try:
            with self._connect() as conn:
                rows = conn.execute("\n".join(sql), params).fetchall()
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to search chunks: {exc}") from exc

        scored: list[SearchResult] = []
        for row in rows:
            score = self._distance_to_score(float(row["distance"]))
            if score < threshold:
                continue
            scored.append(
                SearchResult(
                    repo=str(row["repo_name"]),
                    path=str(row["file_path"]),
                    line=int(row["line_start"]),
                    score=score,
                    snippet=str(row["content"]),
                    lang=str(row["lang"]),
                )
            )

        scored.sort(key=lambda item: (-item.score, item.repo, item.path, item.line))
        return scored[:limit]

    def get_meta(self, key: str) -> str | None:
        self._ensure_initialized()
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to read meta key {key}: {exc}") from exc
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self._ensure_initialized()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO meta(key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, value),
                )
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to write meta key {key}: {exc}") from exc

    def refresh_repo(self, repo_id: int) -> None:
        self._ensure_initialized()
        try:
            with self._connect() as conn:
                self._refresh_repo_stats(conn, {repo_id})
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to refresh repository stats: {exc}") from exc

    def update_repo_stats(self, repo_id: int, file_count: int, chunk_count: int) -> None:
        self._ensure_initialized()
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE repos
                    SET indexed_at = ?, file_count = ?, chunk_count = ?
                    WHERE id = ?
                    """,
                    (now, int(file_count), int(chunk_count), repo_id),
                )
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to update repository stats: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to open database: {exc}") from exc
        conn.row_factory = sqlite3.Row
        try:
            self._load_sqlite_vec(conn)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        except StorageError:
            conn.close()
            raise
        except sqlite3.Error as exc:
            conn.close()
            raise self._storage_error(exc, f"Failed to configure database connection: {exc}") from exc
        return conn

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS repos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        path TEXT UNIQUE NOT NULL,
                        indexed_at TEXT,
                        file_count INTEGER NOT NULL DEFAULT 0,
                        chunk_count INTEGER NOT NULL DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
                        file_path TEXT NOT NULL,
                        line_start INTEGER NOT NULL,
                        line_end INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        lang TEXT,
                        file_mtime REAL NOT NULL,
                        UNIQUE(repo_id, file_path, line_start)
                    );

                    CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);
                    CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(repo_id, file_path);
                    CREATE INDEX IF NOT EXISTS idx_chunks_lang ON chunks(lang);
                    """
                )
                current_dimensions = self._get_meta_value(conn, "embedding_dimensions")
                target_dimensions = (
                    int(current_dimensions) if current_dimensions is not None else DEFAULT_VECTOR_DIMENSIONS
                )
                self._ensure_vec_table(conn, target_dimensions)
                self._set_meta_value(conn, "schema_version", SCHEMA_VERSION)
            self._initialized = True
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to initialize database schema: {exc}") from exc

    def _ensure_embedding_dimensions(self, conn: sqlite3.Connection, dimensions: int) -> None:
        current = self._get_meta_value(conn, "embedding_dimensions")
        vec_dimensions = self._get_vec_dimensions(conn)
        if current is None:
            self._set_meta_value(conn, "embedding_dimensions", str(dimensions))
            if vec_dimensions != dimensions and self._vec_chunk_count(conn) == 0:
                conn.commit()
                self._ensure_vec_table(conn, dimensions, recreate=True)
            return
        if int(current) != dimensions and self._vec_chunk_count(conn) == 0:
            self._set_meta_value(conn, "embedding_dimensions", str(dimensions))
            conn.commit()
            self._ensure_vec_table(conn, dimensions, recreate=True)
            return
        if int(current) != dimensions:
            raise DimensionMismatchError(
                f"Index dimensions {current} do not match current embeddings {dimensions}. "
                "Run index --full."
            )

    def _assert_query_dimensions(self, embedding: Sequence[float]) -> None:
        if not embedding:
            raise StorageError("Query embedding cannot be empty")
        current = self.get_meta("embedding_dimensions")
        if current is None:
            return
        if len(embedding) != int(current):
            raise DimensionMismatchError(
                f"Index dimensions {current} do not match query dimensions {len(embedding)}. "
                "Run index --full."
            )

    def _refresh_repo_stats(self, conn: sqlite3.Connection, repo_ids: Iterable[int]) -> None:
        now = datetime.now(UTC).isoformat()
        for repo_id in sorted(set(repo_ids)):
            row = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT file_path) AS file_count,
                    COUNT(*) AS chunk_count
                FROM chunks
                WHERE repo_id = ?
                """,
                (repo_id,),
            ).fetchone()
            conn.execute(
                """
                UPDATE repos
                SET indexed_at = ?, file_count = ?, chunk_count = ?
                WHERE id = ?
                """,
                (
                    now,
                    int(row["file_count"]) if row else 0,
                    int(row["chunk_count"]) if row else 0,
                    repo_id,
                ),
            )

    def _validate_dimensions(self, embeddings: Sequence[Sequence[float]]) -> int:
        first = len(embeddings[0])
        if first == 0:
            raise StorageError("Embeddings cannot be empty")
        for embedding in embeddings:
            if len(embedding) != first:
                raise StorageError("All embeddings must use the same dimensions")
        return first

    def _has_indexed_chunks(self) -> bool:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS chunk_count FROM vec_chunks").fetchone()
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to inspect index state: {exc}") from exc
        return row is not None and int(row["chunk_count"]) > 0

    def _candidate_count(
        self,
        limit: int,
        repo_id: int | None,
        langs: Sequence[str] | None,
        path_glob: str | None,
    ) -> int:
        if repo_id is None and not langs and path_glob is None:
            return limit
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS chunk_count FROM chunks").fetchone()
        except sqlite3.Error as exc:
            raise self._storage_error(exc, f"Failed to inspect index size: {exc}") from exc
        total = int(row["chunk_count"]) if row is not None else 0
        return max(limit, total)

    def _serialize_query_embedding(self, embedding: Sequence[float]) -> bytes:
        return self._sqlite_vec_serialize_float32()([float(value) for value in embedding])

    def _distance_to_score(self, distance: float) -> float:
        return max(0.0, 1.0 - max(distance, 0.0) / 2.0)

    def _vec_chunk_count(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COUNT(*) AS chunk_count FROM vec_chunks").fetchone()
        return int(row["chunk_count"]) if row is not None else 0

    def _get_vec_dimensions(self, conn: sqlite3.Connection) -> int:
        current = self._get_meta_value(conn, "vec_dimensions")
        if current is not None:
            return int(current)
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
        ).fetchone()
        if row is None or row["sql"] is None:
            return DEFAULT_VECTOR_DIMENSIONS
        match = re.search(r"embedding\s+float\[(\d+)\]", str(row["sql"]), re.IGNORECASE)
        if match is None:
            return DEFAULT_VECTOR_DIMENSIONS
        return int(match.group(1))

    def _ensure_vec_table(
        self,
        conn: sqlite3.Connection,
        dimensions: int,
        recreate: bool = False,
    ) -> None:
        row = conn.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = 'vec_chunks'"
        ).fetchone()
        if row is not None:
            table_type = str(row["type"])
            sql = str(row["sql"] or "")
            is_vec_table = table_type == "table" and "USING VEC0" in sql.upper()
            if is_vec_table and not recreate:
                self._set_meta_value(conn, "vec_dimensions", str(self._get_vec_dimensions(conn)))
                return
            if not is_vec_table:
                has_chunks = conn.execute("SELECT COUNT(*) AS chunk_count FROM chunks").fetchone()
                if has_chunks is not None and int(has_chunks["chunk_count"]) > 0:
                    raise StorageError("Legacy vector index detected. Run index --full to rebuild with sqlite-vec.")
            conn.execute("DROP TABLE IF EXISTS vec_chunks")
            conn.commit()

        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{dimensions}],
                +repo_id INTEGER,
                +lang TEXT,
                +file_path TEXT
            )
            """
        )
        self._set_meta_value(conn, "vec_dimensions", str(dimensions))

    def _load_sqlite_vec(self, conn: sqlite3.Connection) -> None:
        try:
            sqlite_vec = importlib.import_module("sqlite_vec")
        except ModuleNotFoundError as exc:
            raise StorageError("sqlite-vec extension not found. Install: pip install sqlite-vec") from exc

        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (AttributeError, OSError, sqlite3.Error) as exc:
            raise StorageError("sqlite-vec extension not found. Install: pip install sqlite-vec") from exc

    def _sqlite_vec_serialize_float32(self):
        try:
            sqlite_vec = importlib.import_module("sqlite_vec")
        except ModuleNotFoundError as exc:
            raise StorageError("sqlite-vec extension not found. Install: pip install sqlite-vec") from exc
        return sqlite_vec.serialize_float32

    def _get_meta_value(self, conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta_value(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            """
            INSERT INTO meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _row_to_repo(self, row: sqlite3.Row) -> Repo:
        return Repo(
            id=int(row["id"]),
            name=str(row["name"]),
            path=str(row["path"]),
            indexed_at=str(row["indexed_at"]) if row["indexed_at"] is not None else None,
            file_count=int(row["file_count"]),
            chunk_count=int(row["chunk_count"]),
        )

    def _storage_error(self, exc: sqlite3.Error, default_message: str) -> StorageError:
        message = str(exc).lower()
        if "database is locked" in message:
            return StorageError("Database is locked by another process")
        if (
            "database disk image is malformed" in message
            or "file is not a database" in message
            or "malformed" in message
            or "database corrupt" in message
        ):
            return StorageError("Database corrupted. Delete ~/.codesearch/index.db and re-index.")
        return StorageError(default_message)
