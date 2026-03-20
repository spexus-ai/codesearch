from click.testing import CliRunner

from codesearch import __version__
from codesearch.cli import cli


# Test: CLI reports version
# Validates: REQ-747 (global option --version)
def test_cli_version() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


# Test: CLI rejects verbose and quiet together
# Validates: AC-1381 (REQ-748 - mutually exclusive verbose and quiet)
def test_cli_rejects_verbose_and_quiet_together() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["--verbose", "--quiet", "index"])

    assert result.exit_code != 0
    assert "--verbose and --quiet are mutually exclusive" in result.output


# Test: CLI exposes scaffold commands in help output
# Validates: AC-1383 (REQ-747 - global help lists available commands)
def test_cli_help_lists_scaffold_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "repo" in result.output
    assert "index" in result.output
    assert "search" in result.output
    assert "config" in result.output
    assert "mcp" in result.output


def test_cli_help_is_available_for_each_subcommand() -> None:
    runner = CliRunner()

    for command in (["repo", "--help"], ["index", "--help"], ["search", "--help"], ["config", "--help"], ["mcp", "--help"]):
        result = runner.invoke(cli, command)
        assert result.exit_code == 0
        assert "Usage:" in result.output


class KeywordProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                embeddings.append([1.0, 0.0])
            elif "beta" in lowered:
                embeddings.append([0.0, 1.0])
            else:
                embeddings.append([0.5, 0.5])
        return embeddings

    def dimensions(self) -> int:
        return 2


def test_cli_repo_lifecycle(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "index.db"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    add_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    list_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "list"])
    remove_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "remove", "repo-a"])

    assert add_result.exit_code == 0
    assert f'Added repository "repo-a" at {repo_path}' in add_result.output
    assert list_result.exit_code == 0
    assert "name" in list_result.output
    assert "indexed" in list_result.output
    assert "chunks" in list_result.output
    assert "last indexed" in list_result.output
    assert "repo-a" in list_result.output
    assert str(repo_path) in list_result.output
    assert "no" in list_result.output
    assert remove_result.exit_code == 0
    assert 'Removed repository "repo-a" (0 chunks deleted)' in remove_result.output


def test_cli_repo_add_rejects_duplicate_path(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "index.db"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    first = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    duplicate = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path), "--name", "alias"],
    )

    assert first.exit_code == 0
    assert duplicate.exit_code == 1
    assert f'Repository at {repo_path} already registered as "repo-a"' in duplicate.output


def test_cli_index_search_and_config_commands(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "beta.toml").write_text("beta token\n", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "index.db"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    index_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a"])
    status_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "--status"])
    search_result = runner.invoke(
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
            "--format",
            "json",
        ],
    )
    config_set_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "config", "set", "embedding.model", "custom-model"],
    )
    config_get_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "config", "get", "embedding.model"],
    )
    config_show_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "config", "show"],
    )

    assert index_result.exit_code == 0
    assert "Indexed 2 files (2 chunks) in repo-a [" in index_result.output
    assert status_result.exit_code == 0
    assert "repo-a" in status_result.output
    assert "db size" in status_result.output
    assert "2" in status_result.output
    assert search_result.exit_code == 0
    assert '"path": "alpha.toml"' in search_result.output
    assert config_set_result.exit_code == 0
    assert "Warning: changing embedding.model may require re-indexing existing data." in config_set_result.output
    assert config_get_result.exit_code == 0
    assert "custom-model" in config_get_result.output
    assert config_show_result.exit_code == 0
    assert "model = custom-model" in config_show_result.output


def test_cli_index_missing_repo_uses_contract_error(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "index.db"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "missing"])

    assert result.exit_code == 1
    assert 'Repository not found. Use "codesearch repo add" first.' in result.output


def test_cli_config_rejects_invalid_key_and_supports_alt_config_path(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    alt_config_path = tmp_path / "alt" / "config.toml"
    db_path = tmp_path / "index.db"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    invalid_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(alt_config_path), "config", "set", "unknown.key", "value"],
    )
    set_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(alt_config_path), "config", "set", "embedding.provider", "ollama"],
    )
    get_result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(alt_config_path), "config", "get", "embedding.provider"],
    )

    assert invalid_result.exit_code == 1
    assert "Unknown config key: unknown.key" in invalid_result.output
    assert set_result.exit_code == 0
    assert "Updated embedding.provider" in set_result.output
    assert get_result.exit_code == 0
    assert "ollama" in get_result.output
    assert alt_config_path.exists()


def test_cli_config_warns_when_provider_changes_with_indexed_data(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "index.db"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a"])

    result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "config", "set", "embedding.provider", "ollama"],
    )

    assert result.exit_code == 0
    assert "Warning: changing embedding.provider may require re-indexing existing data." in result.output


def test_cli_global_options_apply_before_subcommand_and_verbose_prints_timings(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "index.db"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    add_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    result = runner.invoke(
        cli,
        ["--db", str(db_path), "--config", str(config_path), "--verbose", "index", "repo-a"],
    )

    assert add_result.exit_code == 0
    assert result.exit_code == 0
    assert "Indexed 1 files (1 chunks) in repo-a [" in result.output
    assert "scan=" in result.output
    assert "chunk=" in result.output
    assert "embed=" in result.output
    assert "store=" in result.output


def test_cli_quiet_suppresses_non_error_stdout(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "index.db"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    add_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    index_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a"])
    assert add_result.exit_code == 0
    assert index_result.exit_code == 0

    commands = [
        ["--db", str(db_path), "--config", str(config_path), "--quiet", "repo", "list"],
        ["--db", str(db_path), "--config", str(config_path), "--quiet", "index", "--status"],
        ["--db", str(db_path), "--config", str(config_path), "--quiet", "search", "alpha query", "--format", "json"],
        ["--db", str(db_path), "--config", str(config_path), "--quiet", "config", "show"],
        ["--db", str(db_path), "--config", str(config_path), "--quiet", "config", "get", "embedding.model"],
    ]

    for command in commands:
        result = runner.invoke(cli, command)
        assert result.exit_code == 0
        assert result.output == ""
