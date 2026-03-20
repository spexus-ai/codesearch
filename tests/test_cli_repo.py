from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from codesearch.cli import cli


def test_repo_add_success(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    runner = CliRunner()

    result = runner.invoke(cli, ["--db", str(tmp_path / "index.db"), "repo", "add", str(repo_path)])

    assert result.exit_code == 0
    assert f'Added repository "repo-a" at {repo_path}' in result.output


def test_repo_add_with_alias(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["--db", str(tmp_path / "index.db"), "repo", "add", str(repo_path), "--name", "alias"],
    )

    assert result.exit_code == 0
    assert f'Added repository "alias" at {repo_path}' in result.output


def test_repo_add_duplicate(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    runner = CliRunner()
    runner.invoke(cli, ["--db", str(tmp_path / "index.db"), "repo", "add", str(repo_path)])

    result = runner.invoke(
        cli,
        ["--db", str(tmp_path / "index.db"), "repo", "add", str(repo_path), "--name", "alias"],
    )

    assert result.exit_code == 1
    assert f'Repository at {repo_path} already registered as "repo-a"' in result.output


def test_repo_add_nonexistent(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    runner = CliRunner()

    result = runner.invoke(cli, ["--db", str(tmp_path / "index.db"), "repo", "add", str(missing)])

    assert result.exit_code == 1
    assert f"Directory not found: {missing}" in result.output


def test_repo_add_file_not_dir(tmp_path: Path) -> None:
    file_path = tmp_path / "repo.txt"
    file_path.write_text("x", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(cli, ["--db", str(tmp_path / "index.db"), "repo", "add", str(file_path)])

    assert result.exit_code == 1
    assert f"Not a directory: {file_path}" in result.output


def test_repo_list(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    runner = CliRunner()
    runner.invoke(cli, ["--db", str(tmp_path / "index.db"), "repo", "add", str(repo_path)])

    result = runner.invoke(cli, ["--db", str(tmp_path / "index.db"), "repo", "list"])

    assert result.exit_code == 0
    assert "name" in result.output
    assert "path" in result.output
    assert "indexed" in result.output
    assert "chunks" in result.output
    assert "last indexed" in result.output
    assert "repo-a" in result.output
    assert str(repo_path) in result.output


def test_repo_remove(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    runner = CliRunner()
    runner.invoke(cli, ["--db", str(tmp_path / "index.db"), "repo", "add", str(repo_path)])

    result = runner.invoke(cli, ["--db", str(tmp_path / "index.db"), "repo", "remove", "repo-a"])

    assert result.exit_code == 0
    assert 'Removed repository "repo-a" (0 chunks deleted)' in result.output


def test_repo_remove_not_found(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["--db", str(tmp_path / "index.db"), "repo", "remove", "missing"])

    assert result.exit_code == 1
    assert 'Repository "missing" not found' in result.output
