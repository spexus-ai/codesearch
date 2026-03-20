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
    monkeypatch.chdir(repo_path)

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
    monkeypatch.chdir(repo_path)

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
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    add_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)],
    )
    assert add_result.exit_code == 0
    monkeypatch.chdir(repo_path)

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


# Validates: AC-1398 (REQ-749, REQ-750 - CWD in a repo subdirectory filters results to that subdirectory)
def test_cli_search_autodetects_repo_and_subdir_path_glob(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "src").mkdir()
    (repo_path / "tests").mkdir()
    (repo_path / "src" / "alpha.py").write_text("def alpha():\n    return 'alpha token'\n", encoding="utf-8")
    (repo_path / "tests" / "alpha_test.py").write_text("def alpha_test():\n    return 'alpha token'\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)
    monkeypatch.chdir(repo_path / "src")

    result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "search",
            "alpha query",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload
    assert all(item["path"].startswith("src/") for item in payload)


# Validates: AC-1399 (REQ-749, REQ-753 - CWD at repo root searches the whole repository)
def test_cli_search_autodetects_repo_root_without_path_filter(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "src").mkdir()
    (repo_path / "tests").mkdir()
    (repo_path / "src" / "alpha.py").write_text("def alpha():\n    return 'alpha src'\n", encoding="utf-8")
    (repo_path / "tests" / "alpha_test.py").write_text("def alpha_test():\n    return 'alpha tests'\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)
    monkeypatch.chdir(repo_path)

    result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "search",
            "alpha query",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    paths = {item["path"] for item in payload}
    assert "src/alpha.py" in paths
    assert "tests/alpha_test.py" in paths


# Validates: AC-1400 (REQ-751 - CWD outside registered repositories returns an informational message)
def test_cli_search_reports_no_repo_for_cwd(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    outside_path = tmp_path / "outside"
    outside_path.mkdir()
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)
    monkeypatch.chdir(outside_path)

    result = runner.invoke(
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
        ],
    )

    assert result.exit_code == 0
    assert "No registered repository found for current directory." in result.output


# Validates: AC-1401 (REQ-752 - explicit --repo bypasses CWD autodetection)
def test_cli_search_ignores_cwd_when_repo_is_provided(tmp_path: Path, monkeypatch) -> None:
    repo_a_path = tmp_path / "repo-a"
    repo_a_path.mkdir()
    (repo_a_path / "src").mkdir()
    (repo_a_path / "src" / "alpha.py").write_text("alpha token\n", encoding="utf-8")
    repo_b_path = tmp_path / "repo-b"
    repo_b_path.mkdir()
    (repo_b_path / "beta.py").write_text("beta token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    add_repo_a_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_a_path)],
    )
    add_repo_b_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_b_path)],
    )
    index_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index"])
    assert add_repo_a_result.exit_code == 0
    assert add_repo_b_result.exit_code == 0
    assert index_result.exit_code == 0
    monkeypatch.chdir(repo_a_path / "src")

    result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "search",
            "beta query",
            "--repo",
            "repo-b",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload
    assert all(item["repo"] == "repo-b" for item in payload)
    assert all(item["path"] == "beta.py" for item in payload)


# Validates: AC-1402 (REQ-749, REQ-750 - JSON output keeps paths relative to the repo root under CWD filtering)
def test_cli_search_json_paths_remain_repo_relative_under_cwd_filtering(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "src").mkdir()
    (repo_path / "tests").mkdir()
    (repo_path / "src" / "main.py").write_text("def alpha():\n    return 'alpha token'\n", encoding="utf-8")
    (repo_path / "tests" / "helper.py").write_text("def helper():\n    return 'alpha token'\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _bootstrap_index(runner, db_path, config_path, repo_path)
    monkeypatch.chdir(repo_path / "src")

    result = runner.invoke(
        cli,
        [
            "--db",
            str(db_path),
            "--config",
            str(config_path),
            "search",
            "alpha query",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload
    assert {item["path"] for item in payload} == {"src/main.py"}
