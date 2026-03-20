from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from codesearch.cli import cli


def _connect_with_vec(db_path: Path) -> sqlite3.Connection:
    import sqlite_vec

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class KeywordProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                embeddings.append([1.0, 0.0, 0.0])
            elif "gamma" in lowered:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])
        return embeddings

    def dimensions(self) -> int:
        return 3


def test_e2e_repo_add_index_search_reindex_and_remove(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "sample-repo"
    repo_path.mkdir()
    (repo_path / "alpha.py").write_text("def alpha_feature():\n    return 'alpha token'\n", encoding="utf-8")
    (repo_path / "service.java").write_text("class Service { String id() { return \"beta token\"; } }\n", encoding="utf-8")
    (repo_path / "worker.go").write_text("package main\nfunc Worker() string { return \"beta token\" }\n", encoding="utf-8")
    (repo_path / "config.yaml").write_text("name: beta token\n", encoding="utf-8")
    (repo_path / "settings.toml").write_text("mode = \"beta token\"\n", encoding="utf-8")

    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    add_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    index_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index"])
    monkeypatch.chdir(repo_path)
    search_alpha_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "search", "alpha query", "--format", "json"],
    )

    assert add_result.exit_code == 0
    assert index_result.exit_code == 0
    assert "Indexed 5 files (5 chunks) in sample-repo [" in index_result.output
    alpha_payload = json.loads(search_alpha_result.output)
    assert any(item["path"] == "alpha.py" for item in alpha_payload)

    worker_path = repo_path / "worker.go"
    worker_path.write_text("package main\nfunc Worker() string { return \"gamma token\" }\n", encoding="utf-8")
    os.utime(worker_path, (200.0, 200.0))

    reindex_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index"])
    search_gamma_result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "search",
            "gamma query",
            "--format",
            "json",
        ],
    )
    remove_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "repo", "remove", "sample-repo"],
    )
    list_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "list"])

    assert reindex_result.exit_code == 0
    assert "Indexed 1 files (1 chunks) in sample-repo [" in reindex_result.output
    gamma_payload = json.loads(search_gamma_result.output)
    assert any(item["path"] == "worker.go" for item in gamma_payload)
    assert remove_result.exit_code == 0
    assert 'Removed repository "sample-repo" (5 chunks deleted)' in remove_result.output
    assert list_result.exit_code == 0
    assert "No repositories registered" in list_result.output

    with _connect_with_vec(db_path) as conn:
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vec_count = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]

    assert chunk_count == 0
    assert vec_count == 0
