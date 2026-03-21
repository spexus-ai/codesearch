from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from codesearch.cli import cli
from codesearch.lsh import SimHashLSH
from codesearch.storage import Chunk, Storage


class MappingProvider:
    def __init__(self, mapping: dict[str, list[float]] | None = None, dimensions: int = 3):
        self.mapping = mapping or {
            "alpha": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
        }
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vector = next((values for key, values in self.mapping.items() if key in lowered), None)
            if vector is None:
                vector = [0.0, 0.0, 1.0]
            if len(vector) < self._dimensions:
                vector = vector + [0.0] * (self._dimensions - len(vector))
            embeddings.append(vector[: self._dimensions])
        return embeddings

    def dimensions(self) -> int:
        return self._dimensions


def _bootstrap_index(
    runner: CliRunner,
    db_path: Path,
    config_path: Path,
    repo_path: Path,
    *,
    name: str = "repo-a",
) -> None:
    add_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path), "--name", name],
    )
    assert add_result.exit_code == 0
    index_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", name])
    assert index_result.exit_code == 0


def _chunk_ids_by_path(db_path: Path) -> dict[str, list[int]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, file_path FROM chunks ORDER BY id ASC").fetchall()
    chunk_ids: dict[str, list[int]] = {}
    for chunk_id, file_path in rows:
        chunk_ids.setdefault(str(file_path), []).append(int(chunk_id))
    return chunk_ids


def _force_shared_band(storage: Storage, chunk_ids: list[int], *, band_idx: int = 99, band_hash: bytes = b"forced") -> None:
    for chunk_id in chunk_ids:
        storage.insert_chunk_bands(chunk_id, [(band_idx, band_hash)])


def test_duplicates_happy_path(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "src").mkdir()
    (repo_path / "src" / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "src" / "alpha_copy.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "src" / "beta.py").write_text("beta token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: MappingProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)

    result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "duplicates",
            "--repo",
            "repo-a",
            "--path",
            "src/*.py",
            "--cross-file-only",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload) == 1
    assert payload[0]["chunk_a"]["path"] == "src/alpha.py"
    assert payload[0]["chunk_b"]["path"] == "src/alpha_copy.py"


def test_duplicates_cwd_autodetect(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "src").mkdir()
    (repo_path / "src" / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "src" / "alpha_copy.py").write_text("alpha token\n", encoding="utf-8")
    outside_path = tmp_path / "outside"
    outside_path.mkdir()
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: MappingProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)
    monkeypatch.chdir(repo_path / "src")

    inside_result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "duplicates",
            "--format",
            "json",
        ],
    )

    monkeypatch.chdir(outside_path)
    outside_result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "duplicates",
            "--format",
            "text",
        ],
    )

    assert inside_result.exit_code == 0
    inside_payload = json.loads(inside_result.output)
    assert len(inside_payload) == 1
    assert all(item["chunk_a"]["path"].startswith("src/") for item in inside_payload)
    assert outside_result.exit_code == 0
    assert "No registered repository found for current directory." in outside_result.output


def test_duplicates_threshold(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "soft.py").write_text("soft token\n", encoding="utf-8")
    (repo_path / "close.py").write_text("close token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    provider = MappingProvider(
        mapping={
            "soft": [1.0, 0.0, 0.0],
            "close": [0.92, 0.392, 0.0],
        }
    )
    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: provider)
    _bootstrap_index(runner, db_path, config_path, repo_path)

    storage = Storage(db_path)
    chunk_ids = _chunk_ids_by_path(db_path)
    _force_shared_band(storage, [chunk_ids["soft.py"][0], chunk_ids["close.py"][0]])

    high_threshold = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "duplicates", "--repo", "repo-a", "--threshold", "0.95"],
    )
    low_threshold = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "duplicates", "--repo", "repo-a", "--threshold", "0.90", "--format", "json"],
    )

    assert high_threshold.exit_code == 0
    assert json.loads(high_threshold.output) == []
    assert low_threshold.exit_code == 0
    assert len(json.loads(low_threshold.output)) == 1


def test_duplicates_cross_file_only(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "alpha_copy.py").write_text("alpha token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: MappingProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)

    storage = Storage(db_path)
    repo_id = storage.list_repos()[0].id
    extra_chunk_id = storage.insert_chunks(
        [
            Chunk(
                repo_id=repo_id,
                file_path="alpha.py",
                line_start=10,
                line_end=11,
                content="alpha token",
                lang="python",
                file_mtime=1.0,
            )
        ],
        [[1.0, 0.0, 0.0]],
    )[0]
    chunk_ids = _chunk_ids_by_path(db_path)
    _force_shared_band(
        storage,
        [chunk_ids["alpha.py"][0], chunk_ids["alpha_copy.py"][0], extra_chunk_id],
    )

    result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "duplicates", "--repo", "repo-a", "--cross-file-only", "--format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload
    assert all(item["chunk_a"]["path"] != item["chunk_b"]["path"] for item in payload)


def test_duplicates_multi_repo(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    for name in ("repo-a", "repo-b", "repo-c"):
        repo_path = tmp_path / name
        repo_path.mkdir()
        (repo_path / "alpha.py").write_text("alpha token\n", encoding="utf-8")
        (repo_path / "alpha_copy.py").write_text("alpha token\n", encoding="utf-8")

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: MappingProvider())
    _bootstrap_index(runner, db_path, config_path, tmp_path / "repo-a", name="repo-a")
    _bootstrap_index(runner, db_path, config_path, tmp_path / "repo-b", name="repo-b")
    _bootstrap_index(runner, db_path, config_path, tmp_path / "repo-c", name="repo-c")

    result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "duplicates", "--repo", "repo-a", "--repo", "repo-b", "--format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload
    repos = {item["chunk_a"]["repo"] for item in payload} | {item["chunk_b"]["repo"] for item in payload}
    assert "repo-c" not in repos
    assert repos <= {"repo-a", "repo-b"}


def test_duplicates_path_filter(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "src" / "services").mkdir(parents=True)
    (repo_path / "tests").mkdir()
    (repo_path / "src" / "services" / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "src" / "services" / "alpha_copy.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "tests" / "alpha_test.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "tests" / "alpha_test_copy.py").write_text("alpha token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: MappingProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)

    result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "duplicates", "--repo", "repo-a", "--path", "src/services/*", "--format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload
    assert all(item["chunk_a"]["path"].startswith("src/services/") for item in payload)
    assert all(item["chunk_b"]["path"].startswith("src/services/") for item in payload)


def test_duplicates_dedup_pairs(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "alpha_copy.py").write_text("alpha token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: MappingProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)

    storage = Storage(db_path)
    chunk_ids = _chunk_ids_by_path(db_path)
    pair_chunk_ids = [chunk_ids["alpha.py"][0], chunk_ids["alpha_copy.py"][0]]
    _force_shared_band(storage, pair_chunk_ids, band_idx=99, band_hash=b"band-1")
    _force_shared_band(storage, pair_chunk_ids, band_idx=100, band_hash=b"band-2")

    result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "duplicates", "--repo", "repo-a", "--format", "json"],
    )

    assert result.exit_code == 0
    assert len(json.loads(result.output)) == 1


def test_duplicates_json_format(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "alpha_copy.py").write_text("alpha token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: MappingProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)

    result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "duplicates", "--repo", "repo-a", "--format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0].keys() == {"chunk_a", "chunk_b", "similarity"}


def test_duplicates_empty_index(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--db", str(tmp_path / "index.db"), "--config", str(tmp_path / "config.toml"), "duplicates", "--format", "text"],
    )

    assert result.exit_code == 0
    assert "No indexed chunks found" in result.output


def test_duplicates_reindex_new_model(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "alpha_copy.py").write_text("alpha token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    provider = MappingProvider(dimensions=3)
    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: provider)
    _bootstrap_index(runner, db_path, config_path, repo_path)

    storage = Storage(db_path)
    original_blob = storage.load_hyperplane_matrix()

    provider._dimensions = 4
    provider.mapping["alpha"] = [1.0, 0.0, 0.0, 0.0]
    provider.mapping["beta"] = [0.0, 1.0, 0.0, 0.0]
    reindex_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "index", "repo-a", "--full"],
    )

    assert reindex_result.exit_code == 0
    updated_blob = storage.load_hyperplane_matrix()
    assert updated_blob is not None
    assert updated_blob != original_blob
    restored = SimHashLSH(dim=4).deserialize_matrix(updated_blob)
    assert restored.shape == (72, 4)
