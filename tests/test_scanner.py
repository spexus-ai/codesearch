import shutil
import subprocess
from pathlib import Path

import pytest

from codesearch.scanner import FileScanner


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_scanner_walk_filters_extensions_dirs_and_large_files(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    scanner = FileScanner()

    _write(repo_path / "src" / "main.py", "print('ok')\n")
    _write(repo_path / "src" / "util.ts", "export const x = 1;\n")
    _write(repo_path / "README.md", "# ignored\n")
    _write(repo_path / "node_modules" / "pkg" / "index.js", "console.log('ignored');\n")
    _write(repo_path / "build" / "generated.py", "print('ignored')\n")
    (repo_path / "large.sql").write_text("x" * (100 * 1024 + 1))

    scanned = scanner.scan(repo_path)

    assert set(scanned) == {"src/main.py", "src/util.ts"}
    assert all(value > 0 for value in scanned.values())


def test_scanner_returns_empty_mapping_for_empty_directory(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    scanner = FileScanner()

    scanned = scanner.scan(repo_path)

    assert scanned == {}


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_scanner_prefers_git_tracked_files_for_git_repositories(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    scanner = FileScanner()

    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    _write(repo_path / "tracked.py", "print('tracked')\n")
    _write(repo_path / "untracked.py", "print('untracked')\n")
    _write(repo_path / "notes.txt", "ignored\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=repo_path, check=True, capture_output=True)

    scanned = scanner.scan(repo_path)

    assert set(scanned) == {"tracked.py"}
