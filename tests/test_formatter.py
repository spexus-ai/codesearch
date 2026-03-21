from __future__ import annotations

import json
from pathlib import Path

from codesearch.duplicates import ChunkInfo, DuplicatePair
from codesearch.formatter import (
    auto_format,
    format_duplicates_json,
    format_duplicates_text,
    format_json,
    format_no_snippet,
    format_table,
    format_text,
)
from codesearch.searcher import SearchResult


class FakeStdout:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def test_format_text_renders_header_and_numbered_snippet() -> None:
    result = SearchResult(
        repo="repo-a",
        path="src/main.py",
        line=10,
        score=0.873,
        snippet="def alpha():\n    return 1",
        lang="python",
    )

    output = format_text([result])

    assert "repo-a:src/main.py:10  (score: 0.87)" in output
    assert "  10 | def alpha():" in output
    assert "  11 |     return 1" in output


def test_format_text_uses_file_context_when_repo_paths_are_available(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo-a"
    file_path = repo_path / "src" / "main.py"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
    result = SearchResult(
        repo="repo-a",
        path="src/main.py",
        line=3,
        score=0.9,
        snippet="line3",
        lang="python",
    )

    output = format_text([result], context=1, repo_paths={"repo-a": repo_path})

    assert "   2 | line2" in output
    assert "   3 | line3" in output
    assert "   4 | line4" in output


def test_format_json_serializes_results() -> None:
    result = SearchResult(
        repo="repo-a",
        path="src/main.py",
        line=10,
        score=0.5,
        snippet="def alpha(): pass",
        lang="python",
    )

    payload = json.loads(format_json([result]))

    assert payload == [
        {
            "repo": "repo-a",
            "path": "src/main.py",
            "line": 10,
            "score": 0.5,
            "snippet": "def alpha(): pass",
            "lang": "python",
        }
    ]


def test_format_no_snippet_renders_compact_rows() -> None:
    result = SearchResult(
        repo="repo-a",
        path="src/main.py",
        line=10,
        score=0.873,
        snippet="def alpha(): pass",
        lang="python",
    )

    output = format_no_snippet([result])

    assert output == "repo-a:src/main.py:10  0.87"


def test_format_text_supports_no_snippet_mode() -> None:
    result = SearchResult(
        repo="repo-a",
        path="src/main.py",
        line=10,
        score=0.873,
        snippet="def alpha(): pass",
        lang="python",
    )

    output = format_text([result], no_snippet=True)

    assert output == "repo-a:src/main.py:10  0.87"


def test_auto_format_switches_between_text_and_json() -> None:
    result = SearchResult(
        repo="repo-a",
        path="src/main.py",
        line=10,
        score=0.873,
        snippet="def alpha(): pass",
        lang="python",
    )

    text_output = auto_format([result], stdout=FakeStdout(is_tty=True))
    json_output = auto_format([result], stdout=FakeStdout(is_tty=False))

    assert "repo-a:src/main.py:10  (score: 0.87)" in text_output
    assert json.loads(json_output)[0]["repo"] == "repo-a"


def test_format_table_aligns_columns() -> None:
    rows = [
        {"name": "repo-a", "files": 10, "chunks": 42},
        {"name": "repo-long", "files": 2, "chunks": 5},
    ]

    output = format_table(rows, columns=[("Name", "name"), ("Files", "files"), ("Chunks", "chunks")])

    assert "Name       " in output
    assert "repo-a" in output
    assert "repo-long" in output


def test_format_duplicates_text_and_json() -> None:
    pair = DuplicatePair(
        chunk_a=ChunkInfo(
            repo="repo-a",
            path="src/a.py",
            line_start=1,
            line_end=2,
            snippet="def alpha():\n    return 1",
            lang="python",
        ),
        chunk_b=ChunkInfo(
            repo="repo-a",
            path="src/b.py",
            line_start=5,
            line_end=6,
            snippet="def beta():\n    return 1",
            lang="python",
        ),
        similarity=0.99,
    )

    text_output = format_duplicates_text([pair])
    json_output = json.loads(format_duplicates_json([pair]))

    assert "repo-a:src/a.py:1-2 <-> repo-a:src/b.py:5-6 (score: 0.99)" in text_output
    assert "A:" in text_output
    assert "B:" in text_output
    assert json_output[0]["chunk_a"]["path"] == "src/a.py"
    assert json_output[0]["chunk_b"]["path"] == "src/b.py"
