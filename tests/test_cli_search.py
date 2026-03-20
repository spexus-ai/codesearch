from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from codesearch.cli import cli


class KeywordProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                embeddings.append([1.0, 0.0, 0.0])
            elif "beta" in lowered:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])
        return embeddings

    def dimensions(self) -> int:
        return 3


def _bootstrap_index(runner: CliRunner, db_path: Path, config_path: Path, repo_path: Path) -> None:
    add_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    assert add_result.exit_code == 0
    index_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a"])
    assert index_result.exit_code == 0


def test_cli_search_supports_filters_and_json_output(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "src").mkdir()
    (repo_path / "src" / "alpha.py").write_text("def alpha():\n    return 'alpha token'\n", encoding="utf-8")
    (repo_path / "src" / "beta.ts").write_text("export const beta = () => 'beta token';\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)

    result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "search",
            "alpha query",
            "--repo",
            "repo-a",
            "--lang",
            "python",
            "--path",
            "src/*.py",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload) == 1
    assert payload[0]["path"] == "src/alpha.py"
    assert payload[0]["lang"] == "python"


def test_cli_search_supports_text_context_and_no_snippet(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    file_path = repo_path / "alpha.py"
    file_path.write_text("line1\nalpha token\nline3\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)

    text_result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "search",
            "alpha query",
            "--format",
            "text",
            "--context",
            "1",
        ],
    )
    compact_result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "search",
            "alpha query",
            "--no-snippet",
        ],
    )

    assert text_result.exit_code == 0
    assert "repo-a:alpha.py:1  (score:" in text_result.output
    assert "   1 | line1" in text_result.output
    assert "   2 | alpha token" in text_result.output
    assert compact_result.exit_code == 0
    assert "repo-a:alpha.py:1  " in compact_result.output
    assert "score" not in compact_result.output


def test_cli_search_reports_no_results_for_text_and_json(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)

    text_result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "search",
            "gamma query",
            "--format",
            "text",
            "--threshold",
            "0.3",
        ],
    )
    json_result = runner.invoke(
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
            "--threshold",
            "0.3",
        ],
    )

    assert text_result.exit_code == 0
    assert "No results found" in text_result.output
    assert json.loads(json_result.output) == []


def test_cli_search_reports_empty_index_for_text_and_json(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    text_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "search", "alpha query", "--format", "text"],
    )
    json_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "search", "alpha query", "--format", "json"],
    )

    assert text_result.exit_code == 0
    assert "Index is empty. Run codesearch index first." in text_result.output
    assert json.loads(json_result.output) == []
