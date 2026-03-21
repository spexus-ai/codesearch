# codesearch

Semantic code search CLI powered by embeddings and sqlite-vec.

AI coding agents (Claude Code, Codex, Cursor, etc.) spend a significant portion of their token budget just navigating the codebase — running grep, reading files, and guessing where the relevant code lives. codesearch solves this by pre-indexing your repositories into a vector database. Instead of scanning files one by one, the agent queries the index and gets relevant code chunks in one call — saving tokens and time.

Works as a standalone CLI for developers and integrates with any AI coding agent via MCP protocol or pipe. Agent-agnostic: Claude Code, Codex, OpenCode, and others.

## Features

- **Semantic search** across multiple repositories with cosine similarity ranking
- **AST-aware chunking** via tree-sitter (Python, Java, Go, TypeScript, Rust, C/C++, Ruby, PHP, Kotlin, Scala, C#) with line-based fallback
- **Incremental indexing** — only changed files are re-indexed
- **Multiple embedding providers** — local (sentence-transformers + ONNX) or remote (OpenAI, Ollama, any OpenAI-compatible API)
- **MCP server** for integration with AI coding agents
- **Duplicate detection** — find semantic code duplicates within and across repositories via LSH
- **Pipe-friendly** — JSON output for scripting, text output for humans

## Installation

### One-liner via pipx (recommended)

```bash
# Recommended: language grammars + local embeddings + MCP
pipx install "codesearch[languages,mcp] @ git+https://github.com/spexus-ai/codesearch.git"

# Full: + Ollama and OpenAI providers
pipx install "codesearch[languages,ollama,openai,mcp] @ git+https://github.com/spexus-ai/codesearch.git"

# Minimal: local embeddings only, no grammars
pipx install "codesearch @ git+https://github.com/spexus-ai/codesearch.git"
```

Requires [pipx](https://pipx.pypa.io/).

Available extras:
- `languages` — smart chunking by functions/classes for 13 languages (without it, code is split by lines)
- `ollama` — support for [Ollama](https://ollama.com) as embedding provider
- `openai` — support for OpenAI and OpenAI-compatible APIs (Jina, Together, etc.)
- `mcp` — MCP server for integration with AI coding agents (Claude Code, Codex, etc.)

### From source (for development)

```bash
git clone git@github.com:spexus-ai/codesearch.git && cd codesearch
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"    # includes dev dependencies (pytest)
```

## Quick Start

```bash
# 1. Register a repository
codesearch repo add /path/to/your/repo

# 2. Index it (first run downloads the embedding model ~90MB)
codesearch index

# 3. Search
codesearch search "kafka consumer handler"
```

## Usage

### Repository management

```bash
codesearch repo add /path/to/repo              # register with auto-name
codesearch repo add /path/to/repo --name myapp  # register with alias
codesearch repo list                            # show all repos with stats
codesearch repo remove myapp                    # unregister and delete chunks
```

### Indexing

```bash
codesearch index                  # index all registered repos (incremental)
codesearch index myapp            # index specific repo
codesearch index --full           # force full re-index
codesearch index --status         # show index stats
```

### Search

```bash
codesearch search "authentication middleware"
codesearch search "query" --repo myapp          # filter by repo
codesearch search "query" --lang python --lang go  # filter by language
codesearch search "query" --path "src/api/*"    # filter by file glob
codesearch search "query" --limit 5             # limit results
codesearch search "query" --threshold 0.5       # minimum similarity
codesearch search "query" --format json         # JSON output
codesearch search "query" --format compact      # path:line score only
codesearch search "query" --context 3           # show 3 lines of context
codesearch search "query" --no-snippet          # no code snippets
```

Output format is auto-detected: text for TTY, JSON for pipes.

```bash
# Pipe to jq
codesearch search "error handling" | jq '.[].path'
```

### Duplicate detection

```bash
codesearch duplicates                             # find duplicates (auto-detect repo from CWD)
codesearch duplicates --repo myapp                # within a single repo
codesearch duplicates --repo myapp --cross-file-only  # only cross-file duplicates
codesearch duplicates --repo app --repo lib       # across two repos
codesearch duplicates --path "*/utils/*"           # filter by file path glob
codesearch duplicates --threshold 0.90            # stricter similarity (default: 0.82)
codesearch duplicates --limit 100                 # more results (default: 50)
codesearch duplicates --format json               # JSON output
```

### Configuration

```bash
codesearch config show                         # display current config
codesearch config get embedding.model          # get single value
codesearch config set embedding.model all-MiniLM-L6-v2  # set value
codesearch config set embedding.provider ollama
```

Config file: `~/.codesearch/config.toml` (created on first run).

### Global options

```bash
codesearch --db /custom/path.db ...            # custom database path
codesearch --config /custom/config.toml ...    # custom config path
codesearch -v ...                              # verbose output (timing, tracebacks)
codesearch -q ...                              # quiet (errors only)
codesearch --version                           # show version
```

## Embedding Providers

### Local (default)

Uses `sentence-transformers` with ONNX backend. No API key needed.

```toml
# ~/.codesearch/config.toml
[embedding]
provider = "sentence-transformers"
model = "all-MiniLM-L6-v2"
backend = "onnx"
```

### Ollama

```bash
codesearch config set embedding.provider ollama
codesearch config set embedding.model nomic-embed-text
```

Requires Ollama running on `localhost:11434` and `httpx` installed (`pip install httpx`).

### OpenAI

```bash
codesearch config set embedding.provider openai
codesearch config set embedding.model text-embedding-3-small
export CODESEARCH_API_KEY=sk-...
```

### OpenAI-compatible (Jina, Together, etc.)

```bash
codesearch config set embedding.provider openai-compatible
codesearch config set embedding.model jina-embeddings-v3
codesearch config set embedding.base_url https://api.jina.ai/v1
export CODESEARCH_API_KEY=...
```

API key priority: `CODESEARCH_API_KEY` env var > `embedding.api_key` in config.

## MCP Server

Run as an MCP server for AI coding agents:

```bash
codesearch mcp
```

This starts a stdio-based MCP server exposing the `semantic_search` tool. Requires at least one indexed repository.

### Claude Code integration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "codesearch": {
      "command": "/path/to/.venv/bin/codesearch",
      "args": ["mcp"]
    }
  }
}
```

## Supported Languages

AST-aware chunking (via tree-sitter):

Python, Java, JavaScript, TypeScript, Go, Rust, Ruby, PHP, Scala, C, C++, C#, Kotlin

Line-based fallback (all other file types):

YAML, TOML, SQL, Shell, Gradle, XML, and any file with a supported extension.

Supported extensions: `.py`, `.java`, `.js`, `.ts`, `.tsx`, `.go`, `.rs`, `.rb`, `.php`, `.scala`, `.kt`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.gradle`, `.xml`, `.yml`, `.yaml`, `.toml`, `.sql`, `.sh`, `.bash`

## Database

Index is stored in `~/.codesearch/index.db` (SQLite + sqlite-vec). Override with `--db`.

Files larger than 100KB are skipped. For git repos, only tracked files are indexed (respects `.gitignore`).

## Development

```bash
source .venv/bin/activate
pip install -e ".[all]"
python -m pytest tests/ -v
```

## Architecture

```
codesearch/
├── cli.py               # Click CLI
├── config.py            # TOML configuration
├── storage.py           # SQLite + sqlite-vec
├── scanner.py           # File discovery (git ls-files / os.walk)
├── chunker.py           # tree-sitter AST + line fallback
├── indexer.py           # Indexing pipeline orchestrator
├── searcher.py          # Semantic search
├── duplicates.py        # Duplicate detection engine
├── lsh.py               # SimHash LSH for near-duplicate candidate retrieval
├── formatter.py         # Text / JSON / compact output
├── errors.py            # Exception hierarchy
├── mcp_server.py        # MCP stdio server
└── embedding/
    ├── base.py          # EmbeddingProvider ABC
    ├── factory.py       # Provider factory
    ├── sentence_transformers.py
    ├── ollama.py
    ├── openai.py
    └── openai_compatible.py
```

## License

APACHE-2.0
