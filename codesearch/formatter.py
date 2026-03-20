from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence


def format_text(
    results: Sequence[object],
    context: int = 0,
    no_snippet: bool = False,
    context_lines: int | None = None,
    repo_paths: dict[str, str | Path] | None = None,
) -> str:
    if no_snippet:
        return format_no_snippet(results)

    resolved_context = context if context_lines is None else context_lines
    blocks: list[str] = []
    for result in results:
        blocks.append(_format_result_block(result, context_lines=resolved_context, repo_paths=repo_paths))
    return "\n\n".join(blocks)


def format_json(results: Sequence[object]) -> str:
    payload = [_row_to_mapping(result) for result in results]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_no_snippet(results: Sequence[object]) -> str:
    lines: list[str] = []
    for result in results:
        row = _row_to_mapping(result)
        lines.append(f"{row['repo']}:{row['path']}:{row['line']}  {float(row['score']):.2f}")
    return "\n".join(lines)


def auto_format(
    results: Sequence[object],
    *,
    context_lines: int = 0,
    json_output: bool | None = None,
    no_snippet: bool = False,
    repo_paths: dict[str, str | Path] | None = None,
    stdout: Any | None = None,
) -> str:
    if no_snippet:
        return format_no_snippet(results)

    stream = sys.stdout if stdout is None else stdout
    if json_output is None:
        json_output = not bool(stream.isatty())

    if json_output:
        return format_json(results)
    return format_text(results, context_lines=context_lines, repo_paths=repo_paths)


def format_table(rows: Sequence[object], columns: Sequence[tuple[str, str]] | None = None) -> str:
    if not rows:
        return ""

    normalized = [_row_to_mapping(row) for row in rows]
    if columns is None:
        columns = [(key, key) for key in normalized[0].keys()]

    widths: dict[str, int] = {}
    for header, key in columns:
        widths[key] = max(len(header), *(len(_stringify(row.get(key, ""))) for row in normalized))

    header = "  ".join(header.ljust(widths[key]) for header, key in columns)
    separator = "  ".join("-" * widths[key] for _, key in columns)
    body = [
        "  ".join(_stringify(row.get(key, "")).ljust(widths[key]) for _, key in columns)
        for row in normalized
    ]
    return "\n".join([header, separator, *body])


def _format_result_block(
    result: object,
    *,
    context_lines: int,
    repo_paths: dict[str, str | Path] | None,
) -> str:
    row = _row_to_mapping(result)
    header = f"{row['repo']}:{row['path']}:{row['line']}  (score: {float(row['score']):.2f})"
    snippet = _resolve_snippet(row, context_lines=context_lines, repo_paths=repo_paths)
    return f"{header}\n{snippet}" if snippet else header


def _resolve_snippet(
    row: dict[str, Any],
    *,
    context_lines: int,
    repo_paths: dict[str, str | Path] | None,
) -> str:
    snippet = str(row.get("snippet", "") or "")
    if context_lines > 0 and repo_paths:
        repo_root = repo_paths.get(str(row["repo"]))
        if repo_root is not None:
            try:
                file_path = Path(repo_root) / str(row["path"])
                source_lines = file_path.read_text(encoding="utf-8").splitlines()
                snippet_line_count = max(1, len(snippet.splitlines()) or 1)
                line_start = max(1, int(row["line"]) - context_lines)
                line_end = min(len(source_lines), int(row["line"]) + snippet_line_count - 1 + context_lines)
                excerpt = source_lines[line_start - 1 : line_end]
                return _format_numbered_lines(excerpt, start_line=line_start)
            except OSError:
                pass

    snippet_lines = snippet.splitlines() or [snippet]
    return _format_numbered_lines(snippet_lines, start_line=int(row["line"]))


def _format_numbered_lines(lines: Sequence[str], *, start_line: int) -> str:
    return "\n".join(f"{line_no:>4} | {line}" for line_no, line in enumerate(lines, start=start_line))


def _row_to_mapping(row: object) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if is_dataclass(row):
        return asdict(row)
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    fields = ("repo", "path", "line", "score", "snippet", "lang")
    return {field: getattr(row, field) for field in fields if hasattr(row, field)}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
