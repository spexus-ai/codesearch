from __future__ import annotations

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


def _add_repo(runner: CliRunner, db_path: Path, config_path: Path, repo_path: Path) -> None:
    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    assert result.exit_code == 0


def test_cli_index_indexes_selected_repository(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _add_repo(runner, db_path, config_path, repo_path)

    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a"])

    assert result.exit_code == 0
    assert "Indexed 1 files (1 chunks) in repo-a [" in result.output


def test_cli_index_full_reindexes_all_files(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "beta.toml").write_text("beta token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _add_repo(runner, db_path, config_path, repo_path)
    first = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a"])
    assert first.exit_code == 0

    (repo_path / "alpha.toml").write_text("alpha token updated\n", encoding="utf-8")

    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a", "--full"])

    assert result.exit_code == 0
    assert "Indexed 2 files (2 chunks) in repo-a [" in result.output


def test_cli_index_indexes_all_registered_repositories(tmp_path: Path, monkeypatch) -> None:
    first_repo = tmp_path / "repo-a"
    second_repo = tmp_path / "repo-b"
    first_repo.mkdir()
    second_repo.mkdir()
    (first_repo / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    (second_repo / "beta.toml").write_text("beta token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _add_repo(runner, db_path, config_path, first_repo)
    _add_repo(runner, db_path, config_path, second_repo)

    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index"])

    assert result.exit_code == 0
    assert "Indexed 1 files (1 chunks) in repo-a [" in result.output
    assert "Indexed 1 files (1 chunks) in repo-b [" in result.output


def test_cli_index_status_shows_repo_stats_and_db_size(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    _add_repo(runner, db_path, config_path, repo_path)
    index_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a"])
    assert index_result.exit_code == 0

    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "--status"])

    assert result.exit_code == 0
    assert "repo" in result.output
    assert "files" in result.output
    assert "chunks" in result.output
    assert "db size" in result.output
    assert "last indexed" in result.output
    assert "repo-a" in result.output


def test_cli_index_rejects_unregistered_repository(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())

    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "missing"])

    assert result.exit_code == 1
    assert 'Repository not found. Use "codesearch repo add" first.' in result.output


def test_cli_index_uses_progress_bar_for_tty_stdout(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    (repo_path / "beta.toml").write_text("beta token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()
    bars: list[dict] = []

    class FakeStdout:
        def isatty(self) -> bool:
            return True

    class FakeProgressBar:
        def __init__(self, *, length: int, label: str):
            bars.append({"length": length, "label": label, "updates": []})

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, amount: int) -> None:
            bars[-1]["updates"].append(amount)

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    monkeypatch.setattr("codesearch.cli.click.get_text_stream", lambda name: FakeStdout())
    monkeypatch.setattr("codesearch.cli.click.progressbar", lambda length, label: FakeProgressBar(length=length, label=label))
    _add_repo(runner, db_path, config_path, repo_path)

    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a"])

    assert result.exit_code == 0
    assert len(bars) == 3
    assert bars[0]["label"] == "Chunking repo-a"
    assert bars[0]["length"] == 2
    assert bars[0]["updates"] == [1, 1]
    assert bars[1]["label"] == "Embedding repo-a"
    assert bars[2]["label"] == "Storing repo-a"
