from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

from codesearch.errors import ChunkerError

LANGUAGE_EXTENSIONS = {
    ".java": "java",
    ".kt": "kotlin",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".gradle": "groovy",
    ".xml": "xml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
}

LANGUAGE_NODE_TYPES = {
    "java": {"class_declaration", "method_declaration", "interface_declaration", "enum_declaration"},
    "kotlin": {"class_declaration", "function_declaration", "object_declaration"},
    "python": {"class_definition", "function_definition"},
    "javascript": {"class_declaration", "function_declaration", "method_definition"},
    "typescript": {"class_declaration", "function_declaration", "method_definition"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "ruby": {"method", "class", "module"},
    "php": {"class_declaration", "method_declaration", "function_definition"},
    "scala": {"class_definition", "object_definition", "function_definition"},
    "c_sharp": {"class_declaration", "method_declaration", "struct_declaration"},
}

TREE_SITTER_MODULES = {
    "java": "tree_sitter_java",
    "kotlin": "tree_sitter_kotlin",
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "ruby": "tree_sitter_ruby",
    "php": "tree_sitter_php",
    "scala": "tree_sitter_scala",
    "c_sharp": "tree_sitter_c_sharp",
}

IMPORT_PREFIXES = {
    "java": ("package ", "import "),
    "kotlin": ("package ", "import "),
    "python": ("import ", "from "),
    "javascript": ("import ",),
    "typescript": ("import ",),
    "go": ("package ", "import "),
    "rust": ("use ", "mod ", "#![", "extern crate "),
    "ruby": ("require ", "require_relative "),
    "php": ("<?php", "use ", "require", "include"),
    "scala": ("package ", "import "),
    "groovy": ("package ", "import "),
    "c": ("#include",),
    "cpp": ("#include",),
    "c_sharp": ("using ", "namespace "),
}


@dataclass(slots=True)
class Chunk:
    file_path: str
    line_start: int
    line_end: int
    content: str
    lang: str


@dataclass(slots=True)
class _Segment:
    line_start: int
    line_end: int
    content: str


class Chunker:
    def __init__(self, max_chunk_lines: int = 80, min_chunk_lines: int = 10, overlap: float = 0.5):
        self.max_chunk_lines = max_chunk_lines
        self.min_chunk_lines = min_chunk_lines
        self.overlap = overlap
        self._parsers: dict[str, Any | None] = {}

    def chunk_file(self, path: Path, repo_path: Path) -> list[Chunk]:
        path = Path(path)
        repo_path = Path(repo_path)
        lang = self.detect_lang(path)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ChunkerError(f"Failed to read chunk source {path}: {exc}") from exc

        relative_path = path.relative_to(repo_path).as_posix()
        parser = self._get_parser(lang)
        if parser is not None:
            try:
                ast_chunks = self._chunk_by_ast(relative_path, source, lang, parser)
                if ast_chunks:
                    return ast_chunks
            except Exception:
                pass
        return self._chunk_by_lines(relative_path, source, lang)

    def detect_lang(self, path: Path) -> str:
        return LANGUAGE_EXTENSIONS.get(path.suffix.lower(), "text")

    def _get_parser(self, lang: str) -> Any | None:
        if lang in self._parsers:
            return self._parsers[lang]
        parser = self._load_parser(lang)
        self._parsers[lang] = parser
        return parser

    def _load_parser(self, lang: str) -> Any | None:
        module_name = TREE_SITTER_MODULES.get(lang)
        if module_name is None:
            return None

        try:
            tree_sitter = importlib.import_module("tree_sitter")
            language_module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            return None

        language = None
        if hasattr(language_module, "language"):
            language = language_module.language()
            language = tree_sitter.Language(language)
        elif hasattr(language_module, "LANGUAGE"):
            language = language_module.LANGUAGE
        if language is None:
            return None

        parser = tree_sitter.Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(language)
        else:
            parser.language = language
        return parser

    def _chunk_by_ast(self, relative_path: str, source: str, lang: str, parser: Any) -> list[Chunk]:
        tree = parser.parse(source.encode("utf-8"))
        root = getattr(tree, "root_node", None)
        if root is None:
            return []

        node_types = LANGUAGE_NODE_TYPES.get(lang, set())
        nodes = self._collect_nodes(root, node_types)
        if not nodes:
            return []

        lines = source.splitlines()
        segments: list[_Segment] = []
        for node in nodes:
            line_start, line_end = self._node_line_range(node)
            if line_start > line_end:
                continue
            node_lines = lines[line_start - 1 : line_end]
            if not node_lines:
                continue
            if line_end - line_start + 1 > self.max_chunk_lines:
                segments.extend(self._line_segments(node_lines, line_start))
            else:
                segments.append(
                    _Segment(
                        line_start=line_start,
                        line_end=line_end,
                        content=self._join_lines(node_lines),
                    )
                )

        if not segments:
            return []

        segments = self._merge_small_segments(segments)
        import_segment = self._extract_import_block(lines, lang)
        if import_segment is not None:
            first_segment = segments[0]
            if import_segment.line_end < first_segment.line_start:
                segments[0] = _Segment(
                    line_start=import_segment.line_start,
                    line_end=first_segment.line_end,
                    content=self._join_nonempty([import_segment.content, first_segment.content]),
                )
            else:
                segments.insert(0, import_segment)

        return [
            Chunk(
                file_path=relative_path,
                line_start=segment.line_start,
                line_end=segment.line_end,
                content=segment.content,
                lang=lang,
            )
            for segment in segments
        ]

    def _chunk_by_lines(self, relative_path: str, source: str, lang: str) -> list[Chunk]:
        lines = source.splitlines()
        if not lines:
            return []
        segments = self._line_segments(lines, 1)
        return [
            Chunk(
                file_path=relative_path,
                line_start=segment.line_start,
                line_end=segment.line_end,
                content=segment.content,
                lang=lang,
            )
            for segment in segments
        ]

    def _collect_nodes(self, root: Any, node_types: set[str]) -> list[Any]:
        collected: list[Any] = []

        def visit(node: Any, inside_relevant: bool) -> None:
            node_type = getattr(node, "type", "")
            is_relevant = node_type in node_types
            if is_relevant and not inside_relevant:
                collected.append(node)
            for child in getattr(node, "children", []) or []:
                visit(child, inside_relevant or is_relevant)

        visit(root, False)
        return sorted(collected, key=lambda node: self._node_line_range(node)[0])

    def _node_line_range(self, node: Any) -> tuple[int, int]:
        start_point = getattr(node, "start_point", (0, 0))
        end_point = getattr(node, "end_point", (0, 0))
        return int(start_point[0]) + 1, int(end_point[0]) + 1

    def _line_segments(self, lines: list[str], base_line: int) -> list[_Segment]:
        if not lines:
            return []
        step = max(1, int(round(self.max_chunk_lines * (1 - self.overlap))))
        segments: list[_Segment] = []
        start = 0
        while start < len(lines):
            end = min(len(lines), start + self.max_chunk_lines)
            chunk_lines = lines[start:end]
            segments.append(
                _Segment(
                    line_start=base_line + start,
                    line_end=base_line + end - 1,
                    content=self._join_lines(chunk_lines),
                )
            )
            if end == len(lines):
                break
            start += step
        return segments

    def _merge_small_segments(self, segments: list[_Segment]) -> list[_Segment]:
        if not segments:
            return []
        merged: list[_Segment] = []
        buffer = segments[0]
        for segment in segments[1:]:
            if self._segment_size(buffer) < self.min_chunk_lines:
                buffer = self._merge_segments(buffer, segment)
                continue
            merged.append(buffer)
            buffer = segment
        if merged and self._segment_size(buffer) < self.min_chunk_lines:
            merged[-1] = self._merge_segments(merged[-1], buffer)
        else:
            merged.append(buffer)
        return merged

    def _merge_segments(self, left: _Segment, right: _Segment) -> _Segment:
        return _Segment(
            line_start=min(left.line_start, right.line_start),
            line_end=max(left.line_end, right.line_end),
            content=self._join_nonempty([left.content, right.content]),
        )

    def _segment_size(self, segment: _Segment) -> int:
        return segment.line_end - segment.line_start + 1

    def _extract_import_block(self, lines: list[str], lang: str) -> _Segment | None:
        prefixes = IMPORT_PREFIXES.get(lang)
        if not prefixes or not lines:
            return None

        collected: list[str] = []
        line_end = 0
        in_group = False
        saw_imports = False
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if in_group:
                collected.append(line)
                line_end = index
                if stripped == ")":
                    in_group = False
                continue
            if not stripped:
                if collected:
                    collected.append(line)
                    line_end = index
                continue
            if stripped.startswith(prefixes):
                collected.append(line)
                line_end = index
                saw_imports = True
                if stripped.endswith("("):
                    in_group = True
                continue
            if lang == "go" and saw_imports and stripped.startswith('"'):
                collected.append(line)
                line_end = index
                continue
            break

        if not saw_imports:
            return None
        return _Segment(line_start=1, line_end=line_end, content=self._join_lines(collected))

    def _join_lines(self, lines: list[str]) -> str:
        return "\n".join(lines).strip("\n")

    def _join_nonempty(self, parts: list[str]) -> str:
        return "\n".join(part for part in parts if part).strip("\n")
