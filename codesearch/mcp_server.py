from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any, BinaryIO

from codesearch import __version__
from codesearch.errors import CodeSearchError

PROTOCOL_VERSION = "2024-11-05"


class CodeSearchMCPServer:
    def __init__(self, searcher):
        self.searcher = searcher

    def run(self, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> None:
        input_stream = stdin or sys.stdin.buffer
        output_stream = stdout or sys.stdout.buffer
        while True:
            message = self._read_message(input_stream)
            if message is None:
                return
            response = self.handle_message(message)
            if response is None:
                continue
            try:
                self._write_message(output_stream, response)
            except BrokenPipeError:
                return

    def handle_message(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        if not isinstance(method, str):
            return self._error_response(message, -32600, "Invalid Request")

        if method == "notifications/initialized":
            return None
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return self._success_response(message, {})
        if method == "initialize":
            return self._success_response(
                message,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {
                            "listChanged": False,
                        }
                    },
                    "serverInfo": {
                        "name": "codesearch",
                        "version": __version__,
                    },
                },
            )
        if method == "tools/list":
            return self._success_response(message, {"tools": [self._semantic_search_tool()]})
        if method == "tools/call":
            return self._handle_tool_call(message)
        if method == "resources/list":
            return self._success_response(message, {"resources": []})
        return self._error_response(message, -32601, f"Method not found: {method}")

    def _handle_tool_call(self, message: Mapping[str, Any]) -> dict[str, Any]:
        params = message.get("params")
        if not isinstance(params, Mapping):
            return self._error_response(message, -32602, "Invalid params")

        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str):
            return self._error_response(message, -32602, "Tool name is required")
        if name != "semantic_search":
            return self._error_response(message, -32601, f"Unknown tool: {name}")
        if not isinstance(arguments, Mapping):
            return self._error_response(message, -32602, "Tool arguments must be an object")

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return self._error_response(message, -32602, "semantic_search.query is required")

        try:
            self._ensure_index_ready()
            results = self.searcher.search(
                query=query,
                repo=self._optional_str(arguments.get("repo")),
                langs=self._optional_langs(arguments.get("lang")),
                path_glob=self._optional_str(arguments.get("path")),
                limit=self._optional_int(arguments.get("limit"), default=10),
                threshold=self._optional_float(arguments.get("threshold"), default=0.0),
            )
        except CodeSearchError as exc:
            return self._error_response(message, -32000, str(exc))
        except ValueError as exc:
            return self._error_response(message, -32602, str(exc))

        payload = [
            {
                "repo": result.repo,
                "path": result.path,
                "line": result.line,
                "score": result.score,
                "snippet": result.snippet,
                "lang": result.lang,
            }
            for result in results
        ]
        return self._success_response(
            message,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload, ensure_ascii=False),
                    }
                ],
                "structuredContent": payload,
                "isError": False,
            },
        )

    def _ensure_index_ready(self) -> None:
        repos = self.searcher.storage.list_repos()
        if not repos or not any(repo.chunk_count > 0 for repo in repos):
            raise CodeSearchError("Index is empty. Run codesearch index first.")

    def _semantic_search_tool(self) -> dict[str, Any]:
        return {
            "name": "semantic_search",
            "description": "Semantic code search across indexed repositories",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic query text"},
                    "repo": {"type": "string", "description": "Optional repository name or path"},
                    "lang": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "Optional language or list of languages",
                    },
                    "path": {"type": "string", "description": "Optional file glob filter"},
                    "limit": {"type": "integer", "minimum": 1, "default": 10},
                    "threshold": {"type": "number", "default": 0.0},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Expected a string value")
        return value

    def _optional_langs(self, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
        raise ValueError("semantic_search.lang must be a string or list of strings")

    def _optional_int(self, value: Any, *, default: int) -> int:
        if value is None:
            return default
        if not isinstance(value, int):
            raise ValueError("Expected an integer value")
        return value

    def _optional_float(self, value: Any, *, default: float) -> float:
        if value is None:
            return default
        if not isinstance(value, (int, float)):
            raise ValueError("Expected a numeric value")
        return float(value)

    def _success_response(self, message: Mapping[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": result,
        }

    def _error_response(self, message: Mapping[str, Any], code: int, text: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": code,
                "message": text,
            },
        }

    def _read_message(self, stream: BinaryIO) -> dict[str, Any] | None:
        headers: dict[str, str] = {}
        while True:
            line = stream.readline()
            if line == b"":
                return None
            if line in {b"\r\n", b"\n"}:
                break
            decoded = line.decode("utf-8").strip()
            if ":" not in decoded:
                continue
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        length_header = headers.get("content-length")
        if length_header is None:
            return None
        body = stream.read(int(length_header))
        return json.loads(body.decode("utf-8"))

    def _write_message(self, stream: BinaryIO, message: Mapping[str, Any]) -> None:
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        stream.write(header)
        stream.write(body)
        stream.flush()
