from __future__ import annotations

import os
import subprocess
from pathlib import Path

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg", ".tiff", ".tif",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac", ".ogg", ".webm",
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".rar", ".zst",
    ".jar", ".war", ".ear", ".class",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".lib",
    ".pyc", ".pyo", ".whl", ".egg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".DS_Store",
}

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    "build",
    "target",
    ".gradle",
    "dist",
    "venv",
    ".venv",
}

MAX_FILE_SIZE_BYTES = 100 * 1024
_BINARY_PROBE_SIZE = 8192


class FileScanner:
    def scan(self, repo_path: Path) -> dict[str, float]:
        repo_path = Path(repo_path).expanduser().resolve()
        if self._is_git_repo(repo_path):
            scanned = self._scan_git(repo_path)
            if scanned:
                return scanned
        return self._scan_walk(repo_path)

    def _is_git_repo(self, path: Path) -> bool:
        return (path / ".git").exists()

    def _scan_git(self, repo_path: Path) -> dict[str, float]:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return {}

        files: dict[str, float] = {}
        for raw_path in result.stdout.decode("utf-8", errors="ignore").split("\x00"):
            if not raw_path:
                continue
            file_path = repo_path / raw_path
            if not self._should_include(file_path):
                continue
            files[Path(raw_path).as_posix()] = file_path.stat().st_mtime
        return files

    def _scan_walk(self, repo_path: Path) -> dict[str, float]:
        files: dict[str, float] = {}
        for root, dirnames, filenames in os.walk(repo_path):
            dirnames[:] = [dirname for dirname in dirnames if dirname not in IGNORE_DIRS]
            root_path = Path(root)
            for filename in filenames:
                file_path = root_path / filename
                if not self._should_include(file_path):
                    continue
                relative_path = file_path.relative_to(repo_path).as_posix()
                files[relative_path] = file_path.stat().st_mtime
        return files

    def _should_include(self, file_path: Path) -> bool:
        if not file_path.is_file():
            return False
        if file_path.suffix.lower() in BINARY_EXTENSIONS:
            return False
        if any(part in IGNORE_DIRS for part in file_path.parts):
            return False
        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
        return not self._looks_binary(file_path)

    @staticmethod
    def _looks_binary(file_path: Path) -> bool:
        try:
            with open(file_path, "rb") as fh:
                chunk = fh.read(_BINARY_PROBE_SIZE)
        except OSError:
            return True
        return b"\x00" in chunk
