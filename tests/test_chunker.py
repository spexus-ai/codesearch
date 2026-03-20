from dataclasses import dataclass, field
from pathlib import Path

from codesearch.chunker import Chunker


@dataclass
class FakeNode:
    type: str
    start_line: int
    end_line: int
    children: list["FakeNode"] = field(default_factory=list)

    @property
    def start_point(self) -> tuple[int, int]:
        return (self.start_line - 1, 0)

    @property
    def end_point(self) -> tuple[int, int]:
        return (self.end_line - 1, 0)


@dataclass
class FakeTree:
    root_node: FakeNode


class FakeParser:
    def __init__(self, root: FakeNode):
        self._root = root

    def parse(self, _: bytes) -> FakeTree:
        return FakeTree(root_node=self._root)


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_chunker_detects_languages() -> None:
    chunker = Chunker()

    assert chunker.detect_lang(Path("main.py")) == "python"
    assert chunker.detect_lang(Path("service.ts")) == "typescript"
    assert chunker.detect_lang(Path("build.gradle")) == "groovy"
    assert chunker.detect_lang(Path("script.sh")) == "bash"
    assert chunker.detect_lang(Path("header.h")) == "cpp"
    assert chunker.detect_lang(Path("Program.cs")) == "c_sharp"
    assert chunker.detect_lang(Path("README.md")) == "text"


def test_chunker_chunks_python_ast_with_import_prepend(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    file_path = repo_path / "app.py"
    _write_file(
        file_path,
        "\n".join(
            [
                "import os",
                "",
                "def alpha():",
                "    x = 1",
                "    y = 2",
                "    return x + y",
                "",
                "def beta():",
                "    return 42",
            ]
        ),
    )
    parser = FakeParser(
        FakeNode(
            "module",
            1,
            9,
            children=[
                FakeNode("function_definition", 3, 6),
                FakeNode("function_definition", 8, 9),
            ],
        )
    )
    chunker = Chunker(min_chunk_lines=1)
    chunker._parsers["python"] = parser

    chunks = chunker.chunk_file(file_path, repo_path)

    assert len(chunks) == 2
    assert chunks[0].lang == "python"
    assert chunks[0].line_start == 1
    assert "import os" in chunks[0].content
    assert "def alpha" in chunks[0].content
    assert "def beta" in chunks[1].content


def test_chunker_chunks_java_ast(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    file_path = repo_path / "src" / "Main.java"
    _write_file(
        file_path,
        "\n".join(
            [
                "package demo;",
                "import java.util.List;",
                "",
                "class Main {",
                "  void alpha() {",
                "    System.out.println(1);",
                "  }",
                "}",
                "",
                "interface Worker {}",
            ]
        ),
    )
    parser = FakeParser(
        FakeNode(
            "program",
            1,
            10,
            children=[
                FakeNode("class_declaration", 4, 8),
                FakeNode("interface_declaration", 10, 10),
            ],
        )
    )
    chunker = Chunker(min_chunk_lines=1)
    chunker._parsers["java"] = parser

    chunks = chunker.chunk_file(file_path, repo_path)

    assert len(chunks) == 2
    assert chunks[0].lang == "java"
    assert chunks[0].line_start == 1
    assert "package demo;" in chunks[0].content
    assert "class Main" in chunks[0].content
    assert "interface Worker" in chunks[1].content


def test_chunker_chunks_go_ast_with_multiline_import_block(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    file_path = repo_path / "main.go"
    _write_file(
        file_path,
        "\n".join(
            [
                "package main",
                "",
                "import (",
                '    "fmt"',
                ")",
                "",
                "func main() {",
                '    fmt.Println("ok")',
                "}",
            ]
        ),
    )
    parser = FakeParser(
        FakeNode(
            "source_file",
            1,
            9,
            children=[FakeNode("function_declaration", 7, 9)],
        )
    )
    chunker = Chunker(min_chunk_lines=1)
    chunker._parsers["go"] = parser

    chunks = chunker.chunk_file(file_path, repo_path)

    assert len(chunks) == 1
    assert chunks[0].lang == "go"
    assert chunks[0].line_start == 1
    assert "import (" in chunks[0].content
    assert "func main()" in chunks[0].content


def test_chunker_chunks_typescript_ast(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    file_path = repo_path / "src" / "main.ts"
    _write_file(
        file_path,
        "\n".join(
            [
                "import { readFileSync } from 'fs';",
                "",
                "function alpha() {",
                "  return readFileSync('x');",
                "}",
                "",
                "class Beta {}",
            ]
        ),
    )
    parser = FakeParser(
        FakeNode(
            "program",
            1,
            7,
            children=[
                FakeNode("function_declaration", 3, 5),
                FakeNode("class_declaration", 7, 7),
            ],
        )
    )
    chunker = Chunker(min_chunk_lines=1)
    chunker._parsers["typescript"] = parser

    chunks = chunker.chunk_file(file_path, repo_path)

    assert len(chunks) == 2
    assert chunks[0].lang == "typescript"
    assert "import { readFileSync }" in chunks[0].content
    assert "function alpha" in chunks[0].content
    assert "class Beta" in chunks[1].content


def test_chunker_falls_back_to_overlapping_line_windows(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    file_path = repo_path / "config.yaml"
    _write_file(file_path, "\n".join(f"line-{idx}" for idx in range(1, 101)))
    chunker = Chunker()

    chunks = chunker.chunk_file(file_path, repo_path)

    assert len(chunks) == 2
    assert chunks[0].lang == "yaml"
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 80
    assert chunks[1].line_start == 41
    assert chunks[1].line_end == 100


def test_chunker_falls_back_to_line_windows_when_parser_errors(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    file_path = repo_path / "broken.py"
    _write_file(file_path, "\n".join(f"line_{idx} = {idx}" for idx in range(1, 101)))

    class BrokenParser:
        def parse(self, _: bytes):
            raise RuntimeError("parse failed")

    chunker = Chunker()
    chunker._parsers["python"] = BrokenParser()

    chunks = chunker.chunk_file(file_path, repo_path)

    assert len(chunks) == 2
    assert chunks[0].lang == "python"
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 80
    assert chunks[1].line_start == 41
    assert chunks[1].line_end == 100
