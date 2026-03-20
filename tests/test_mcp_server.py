from __future__ import annotations

import io
import json

from click.testing import CliRunner

from codesearch.cli import cli
from codesearch.mcp_server import CodeSearchMCPServer
from codesearch.searcher import SearchResult


class FakeSearcher:
    def __init__(self, results: list[SearchResult], *, has_index: bool = True):
        self.results = results
        self.calls: list[dict] = []
        self.storage = _FakeStorage(has_index=has_index)

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class _FakeStorage:
    def __init__(self, *, has_index: bool):
        self.has_index = has_index

    def list_repos(self):
        if not self.has_index:
            return []
        return [_FakeRepo()]


class _FakeRepo:
    chunk_count = 1


class KeywordProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])
        return embeddings

    def dimensions(self) -> int:
        return 2


class BrokenPipeBuffer(io.BytesIO):
    def flush(self) -> None:
        raise BrokenPipeError


def _encode_message(message: dict) -> bytes:
    body = json.dumps(message).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8") + body


def _decode_messages(buffer: bytes) -> list[dict]:
    messages: list[dict] = []
    cursor = 0
    while cursor < len(buffer):
        header_end = buffer.index(b"\r\n\r\n", cursor)
        header_text = buffer[cursor:header_end].decode("utf-8")
        headers = {}
        for line in header_text.split("\r\n"):
            key, value = line.split(":", 1)
            headers[key.lower()] = value.strip()
        cursor = header_end + 4
        length = int(headers["content-length"])
        body = buffer[cursor : cursor + length]
        cursor += length
        messages.append(json.loads(body.decode("utf-8")))
    return messages


def test_mcp_server_handles_initialize_list_and_semantic_search() -> None:
    searcher = FakeSearcher(
        [
            SearchResult(
                repo="repo-a",
                path="src/main.py",
                line=10,
                score=0.9,
                snippet="def alpha(): pass",
                lang="python",
            )
        ]
    )
    server = CodeSearchMCPServer(searcher)
    stdin = io.BytesIO(
        b"".join(
            [
                _encode_message(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "test-client", "version": "1.0.0"},
                        },
                    }
                ),
                _encode_message({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                _encode_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                _encode_message(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "semantic_search",
                            "arguments": {
                                "query": "alpha query",
                                "repo": "repo-a",
                                "lang": ["python"],
                                "path": "src/*.py",
                                "limit": 5,
                                "threshold": 0.7,
                            },
                        },
                    }
                ),
            ]
        )
    )
    stdout = io.BytesIO()

    server.run(stdin=stdin, stdout=stdout)
    responses = _decode_messages(stdout.getvalue())

    assert responses[0]["result"]["serverInfo"]["name"] == "codesearch"
    assert responses[1]["result"]["tools"][0]["name"] == "semantic_search"
    assert searcher.calls == [
        {
            "query": "alpha query",
            "repo": "repo-a",
            "langs": ["python"],
            "path_glob": "src/*.py",
            "limit": 5,
            "threshold": 0.7,
        }
    ]
    payload = responses[2]["result"]["structuredContent"]
    assert payload[0]["path"] == "src/main.py"


def test_mcp_server_returns_error_for_empty_index() -> None:
    server = CodeSearchMCPServer(FakeSearcher([], has_index=False))

    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "semantic_search",
                "arguments": {"query": "alpha"},
            },
        }
    )

    assert response is not None
    assert response["error"]["message"] == "Index is empty. Run codesearch index first."


def test_cli_mcp_requires_non_empty_index(tmp_path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"

    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "mcp"])

    assert result.exit_code == 1
    assert "Index is empty. Run codesearch index first." in result.output


def test_cli_mcp_runs_server_when_index_exists(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "repo-a"
    repo_path.mkdir()
    (repo_path / "alpha.toml").write_text("alpha token\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    config_path = tmp_path / "config.toml"
    runner = CliRunner()
    called = {"run": False}

    monkeypatch.setattr("codesearch.cli.create_provider", lambda config: KeywordProvider())
    add_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "repo", "add", str(repo_path)])
    assert add_result.exit_code == 0
    index_result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "index", "repo-a"])
    assert index_result.exit_code == 0

    def fake_run(self):
        called["run"] = True

    monkeypatch.setattr("codesearch.mcp_server.CodeSearchMCPServer.run", fake_run)

    result = runner.invoke(cli, ["--db", str(db_path), "--config", str(config_path), "mcp"])

    assert result.exit_code == 0
    assert called["run"] is True


def test_mcp_server_gracefully_stops_on_broken_pipe() -> None:
    searcher = FakeSearcher(
        [
            SearchResult(
                repo="repo-a",
                path="src/main.py",
                line=10,
                score=0.9,
                snippet="def alpha(): pass",
                lang="python",
            )
        ]
    )
    server = CodeSearchMCPServer(searcher)
    stdin = io.BytesIO(
        _encode_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            }
        )
    )
    stdout = BrokenPipeBuffer()

    server.run(stdin=stdin, stdout=stdout)
