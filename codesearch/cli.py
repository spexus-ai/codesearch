import functools
import os
import traceback
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import click

from codesearch import __version__
from codesearch.config import ConfigManager
from codesearch.errors import CodeSearchError
from codesearch.formatter import auto_format, format_duplicates_json, format_duplicates_text, format_table
from codesearch.lsh import SimHashLSH

if TYPE_CHECKING:
    from codesearch.storage import Storage


def handle_errors(func):
    """Wrap CLI commands with the project-wide error contract."""

    @functools.wraps(func)
    @click.pass_context
    def wrapper(ctx, *args, **kwargs):
        try:
            return ctx.invoke(func, *args, **kwargs)
        except CodeSearchError as exc:
            click.echo(str(exc), err=True)
            raise SystemExit(1) from exc
        except Exception as exc:  # pragma: no cover - exercised via real runtime failures
            if ctx.obj.get("verbose"):
                traceback.print_exc()
            else:
                click.echo(f"Unexpected error: {exc}", err=True)
            raise SystemExit(2) from exc

    return wrapper


@click.group()
@click.option(
    "--db",
    default="~/.codesearch/index.db",
    show_default=True,
    help="Path to the SQLite index database.",
)
@click.option(
    "--config",
    "config_path",
    default="~/.codesearch/config.toml",
    show_default=True,
    help="Path to the TOML configuration file.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose error output.")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress non-error command output.")
@click.version_option(version=__version__, prog_name="codesearch", message="%(prog)s %(version)s")
@click.pass_context
def cli(ctx, db, config_path, verbose, quiet):
    """Semantic code search via embeddings and sqlite-vec."""
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive")

    ctx.ensure_object(dict)
    ctx.obj["db_path"] = Path(db).expanduser()
    ctx.obj["config_path"] = Path(config_path).expanduser()
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


def _get_storage(ctx: click.Context) -> "Storage":
    from codesearch.storage import Storage

    return Storage(ctx.obj["db_path"])


def _get_config_manager(ctx: click.Context) -> tuple[ConfigManager, object]:
    manager = ConfigManager()
    config = manager.load(ctx.obj["config_path"])
    return manager, config


def _build_provider_config(ctx: click.Context):
    manager, config = _get_config_manager(ctx)
    embedding = config.embedding
    return replace(
        embedding,
        api_key=manager.resolve_api_key(embedding),
    )


def create_provider(config):
    from codesearch.embedding.factory import create_provider

    return create_provider(config)


def _resolve_repo(storage: "Storage", name_or_path: str):
    repo_text = name_or_path.strip()
    repo_path = str(Path(repo_text).expanduser().resolve())
    for repo in storage.list_repos():
        if repo.name == repo_text or repo.path == repo_path:
            return repo
    raise CodeSearchError('Repository not found. Use "codesearch repo add" first.')


def _resolve_repo_from_cwd(storage: "Storage"):
    cwd = Path.cwd().resolve()
    matching_repos = []
    for repo in storage.list_repos():
        repo_path = Path(repo.path).resolve()
        if cwd.is_relative_to(repo_path):
            matching_repos.append((len(repo_path.parts), repo))
    if not matching_repos:
        return None, None
    matching_repos.sort(key=lambda item: item[0], reverse=True)
    selected_repo = matching_repos[0][1]
    repo_root = Path(selected_repo.path).resolve()
    if cwd == repo_root:
        return selected_repo, None
    relative_subdir = cwd.relative_to(repo_root).as_posix()
    return selected_repo, f"{relative_subdir}/*"


def _load_lsh_from_storage(storage: "Storage") -> SimHashLSH:
    dimensions = storage.get_meta("embedding_dimensions")
    if dimensions is None:
        raise CodeSearchError("No indexed chunks found")
    params = storage.load_lsh_params()
    if params is None:
        lsh = SimHashLSH(dim=int(dimensions))
    else:
        num_bands, band_width = params
        lsh = SimHashLSH(num_bands=num_bands, band_width=band_width, dim=int(dimensions))
    matrix_blob = storage.load_hyperplane_matrix()
    if matrix_blob is not None:
        lsh._matrix = lsh.deserialize_matrix(matrix_blob)
    return lsh


def _ensure_directory(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise CodeSearchError(f"Directory not found: {path}")
    if not path.is_dir():
        raise CodeSearchError(f"Not a directory: {path}")
    return path


def _echo_if_not_quiet(ctx: click.Context, text: str) -> None:
    if not ctx.obj["quiet"]:
        click.echo(text)


@cli.group()
def repo():
    """Repository management commands."""


@repo.command("add")
@click.argument("path")
@click.option("--name", default=None, help="Alias for the repository.")
@handle_errors
def repo_add(path, name):
    """Register a repository for indexing."""
    ctx = click.get_current_context()
    storage = _get_storage(ctx)
    repo_path = _ensure_directory(path)
    repo_name = name or repo_path.name
    for repo in storage.list_repos():
        if repo.path == str(repo_path):
            raise CodeSearchError(f'Repository at {repo_path} already registered as "{repo.name}"')
    storage.add_repo(repo_name, str(repo_path))
    if not ctx.obj["quiet"]:
        click.echo(f'Added repository "{repo_name}" at {repo_path}')


@repo.command("list")
@handle_errors
def repo_list():
    """List registered repositories."""
    ctx = click.get_current_context()
    storage = _get_storage(ctx)
    repos = storage.list_repos()
    if not repos:
        _echo_if_not_quiet(ctx, "No repositories registered")
        return
    if ctx.obj["quiet"]:
        return
    rows = [
        {
            "name": repo.name,
            "path": repo.path,
            "indexed": "yes" if repo.indexed_at else "no",
            "chunks": repo.chunk_count,
            "last_indexed": repo.indexed_at or "",
        }
        for repo in repos
    ]
    click.echo(
        format_table(
            rows,
            columns=[
                ("name", "name"),
                ("path", "path"),
                ("indexed", "indexed"),
                ("chunks", "chunks"),
                ("last indexed", "last_indexed"),
            ],
        )
    )


@repo.command("remove")
@click.argument("name")
@handle_errors
def repo_remove(name):
    """Remove a repository and its indexed data."""
    ctx = click.get_current_context()
    storage = _get_storage(ctx)
    repo_path = str(Path(name).expanduser().resolve())
    repo_label = name
    repo_exists = False
    for repo in storage.list_repos():
        if repo.name == name or repo.path == repo_path:
            repo_exists = True
            repo_label = repo.name
            break
    removed = storage.remove_repo(name)
    if removed == 0 and not repo_exists:
        raise CodeSearchError(f'Repository "{name}" not found')
    if not ctx.obj["quiet"]:
        click.echo(f'Removed repository "{repo_label}" ({removed} chunks deleted)')


@cli.command()
@click.argument("target", required=False)
@click.option("--full", is_flag=True, default=False, help="Reindex all files for the selected repositories.")
@click.option("--status", "show_status", is_flag=True, default=False, help="Show repository indexing status.")
@handle_errors
def index(target, full, show_status):
    """Index registered repositories."""
    ctx = click.get_current_context()
    storage = _get_storage(ctx)
    repos = storage.list_repos()
    if show_status:
        if not repos:
            _echo_if_not_quiet(ctx, "No repositories registered.")
            return
        if ctx.obj["quiet"]:
            return
        db_size = os.path.getsize(ctx.obj["db_path"]) if ctx.obj["db_path"].exists() else 0
        rows = [
            {
                "repo": repo.name,
                "files": repo.file_count,
                "chunks": repo.chunk_count,
                "db_size": db_size,
                "last_indexed": repo.indexed_at or "",
            }
            for repo in repos
        ]
        click.echo(
            format_table(
                rows,
                columns=[
                    ("repo", "repo"),
                    ("files", "files"),
                    ("chunks", "chunks"),
                    ("db size", "db_size"),
                    ("last indexed", "last_indexed"),
                ],
            )
        )
        return

    if not repos and target:
        raise CodeSearchError('Repository not found. Use "codesearch repo add" first.')

    if not repos:
        raise CodeSearchError("No repositories registered.")

    selected = [_resolve_repo(storage, target)] if target else repos
    from codesearch.chunker import Chunker
    from codesearch.indexer import Indexer
    from codesearch.scanner import FileScanner

    provider = create_provider(_build_provider_config(ctx))
    scanner = FileScanner()
    chunker = Chunker()
    indexer = Indexer(storage=storage, provider=provider, scanner=scanner, chunker=chunker)

    phase_labels = {
        "chunk": "Chunking",
        "embed": "Embedding",
        "store": "Storing",
    }

    for repo in selected:
        progress_bar = None
        current_phase = None
        last_progress = 0

        def progress_callback(phase: str, done: int, total: int) -> None:
            nonlocal progress_bar, current_phase, last_progress
            if ctx.obj["quiet"] or not click.get_text_stream("stdout").isatty():
                return
            if total <= 0:
                return
            if phase != current_phase:
                if progress_bar is not None:
                    progress_bar.__exit__(None, None, None)
                label = f"{phase_labels.get(phase, phase)} {repo.name}"
                progress_bar = click.progressbar(length=total, label=label)
                progress_bar.__enter__()
                current_phase = phase
                last_progress = 0
            progress_bar.update(done - last_progress)
            last_progress = done

        try:
            result = indexer.index_repo(repo.id, Path(repo.path), full=full, progress_callback=progress_callback)
        finally:
            if progress_bar is not None:
                progress_bar.__exit__(None, None, None)

        if not ctx.obj["quiet"]:
            click.echo(
                f"Indexed {result.files_indexed} files ({result.chunks_created} chunks) "
                f"in {repo.name} [{result.duration_seconds:.1f}s]"
            )
            if ctx.obj["verbose"]:
                click.echo(
                    "  "
                    f"scan={result.scan_seconds:.2f}s "
                    f"chunk={result.chunk_seconds:.2f}s "
                    f"embed={result.embed_seconds:.2f}s "
                    f"lsh={result.lsh_seconds:.2f}s "
                    f"store={result.store_seconds:.2f}s"
                )


@cli.command()
@click.argument("query")
@click.option("--repo", "repo_filter", default=None, help="Restrict search to a repository name or path.")
@click.option("--lang", "langs", multiple=True, help="Restrict search to one or more languages.")
@click.option("--path", "path_glob", default=None, help="Restrict search to files matching a glob.")
@click.option("--limit", default=10, show_default=True, type=int, help="Maximum number of results to return.")
@click.option("--threshold", default=0.0, show_default=True, type=float, help="Minimum similarity threshold.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["auto", "text", "json", "compact"]),
    default="auto",
    show_default=True,
    help="Output format.",
)
@click.option("--context", default=0, show_default=True, type=int, help="Extra context lines around the hit.")
@click.option("--no-snippet", is_flag=True, default=False, help="Print only compact result lines.")
@handle_errors
def search(query, repo_filter, langs, path_glob, limit, threshold, output_format, context, no_snippet):
    """Search indexed code semantically."""
    ctx = click.get_current_context()
    storage = _get_storage(ctx)
    from codesearch.searcher import Searcher

    repos = storage.list_repos()
    prefers_json = output_format == "json" or (
        output_format == "auto" and not click.get_text_stream("stdout").isatty()
    )
    if not repos:
        _echo_if_not_quiet(ctx, "[]" if prefers_json else "Index is empty. Run codesearch index first.")
        return

    if repo_filter is None:
        selected_repo, cwd_path_glob = _resolve_repo_from_cwd(storage)
        if selected_repo is None:
            message = "No registered repository found for current directory."
            if prefers_json:
                if not ctx.obj["quiet"]:
                    click.echo(message, err=True)
                _echo_if_not_quiet(ctx, "[]")
            else:
                _echo_if_not_quiet(ctx, message)
            return
        repo_filter = selected_repo.name
        if path_glob is None:
            path_glob = cwd_path_glob

    has_index = any(repo.chunk_count > 0 for repo in repos)
    if not has_index:
        _echo_if_not_quiet(ctx, "[]" if prefers_json else "Index is empty. Run codesearch index first.")
        return

    provider = create_provider(_build_provider_config(ctx))
    searcher = Searcher(storage=storage, provider=provider)
    results = searcher.search(
        query,
        repo=repo_filter,
        langs=list(langs) or None,
        path_glob=path_glob,
        limit=limit,
        threshold=threshold,
    )
    if not results:
        _echo_if_not_quiet(ctx, "[]" if prefers_json else "No results found")
        return

    if ctx.obj["quiet"]:
        return

    repo_paths = {repo.name: repo.path for repo in repos}
    if output_format == "json":
        rendered = auto_format(results, json_output=True, repo_paths=repo_paths)
    elif output_format == "compact" or no_snippet:
        rendered = auto_format(results, no_snippet=True, repo_paths=repo_paths)
    elif output_format == "text":
        rendered = auto_format(results, json_output=False, context_lines=context, repo_paths=repo_paths)
    else:
        rendered = auto_format(results, context_lines=context, repo_paths=repo_paths)
    click.echo(rendered)


@cli.command()
@click.option("--repo", "repo_filters", multiple=True, help="Filter by repository (repeatable).")
@click.option("--path", "path_filters", multiple=True, help="Filter by file path glob (repeatable).")
@click.option("--cross-file-only", is_flag=True, help="Only return cross-file duplicates.")
@click.option("--threshold", type=float, default=0.82, show_default=True, help="Minimum similarity.")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum number of pairs.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["auto", "text", "json"]),
    default="auto",
    show_default=True,
    help="Output format.",
)
@handle_errors
def duplicates(repo_filters, path_filters, cross_file_only, threshold, limit, output_format):
    """Find semantic duplicate chunks."""
    ctx = click.get_current_context()
    storage = _get_storage(ctx)
    from codesearch.duplicates import DuplicateFinder

    repos = storage.list_repos()
    prefers_json = output_format == "json" or (
        output_format == "auto" and not click.get_text_stream("stdout").isatty()
    )
    if not repos or not any(repo.chunk_count > 0 for repo in repos):
        _echo_if_not_quiet(ctx, "[]" if prefers_json else "No indexed chunks found")
        return

    repo_names = list(repo_filters)
    path_globs = list(path_filters)
    if not repo_names and not path_globs:
        selected_repo, cwd_path_glob = _resolve_repo_from_cwd(storage)
        if selected_repo is None:
            message = "No registered repository found for current directory."
            if prefers_json:
                if not ctx.obj["quiet"]:
                    click.echo(message, err=True)
                _echo_if_not_quiet(ctx, "[]")
            else:
                _echo_if_not_quiet(ctx, message)
            return
        repo_names = [selected_repo.name]
        if cwd_path_glob is not None:
            path_globs = [cwd_path_glob]

    repo_ids = [_resolve_repo(storage, item).id for item in repo_names] or None

    if not storage.has_chunk_bands():
        _echo_if_not_quiet(
            ctx,
            "[]" if prefers_json else "No LSH bands found. Re-index with: codesearch index --full",
        )
        return

    finder = DuplicateFinder(
        storage=storage,
        lsh=_load_lsh_from_storage(storage),
    )
    pairs = finder.find_duplicates(
        repo_ids=repo_ids,
        path_globs=path_globs or None,
        cross_file_only=cross_file_only,
        threshold=threshold,
        limit=limit,
    )
    if not pairs:
        _echo_if_not_quiet(ctx, "[]" if prefers_json else "No duplicates found")
        return
    if ctx.obj["quiet"]:
        return
    if output_format == "json" or (output_format == "auto" and prefers_json):
        click.echo(format_duplicates_json(pairs))
        return
    click.echo(format_duplicates_text(pairs))


@cli.group("config")
def config_group():
    """Configuration commands."""


@config_group.command("show")
@handle_errors
def config_show():
    """Show current configuration."""
    ctx = click.get_current_context()
    manager, config = _get_config_manager(ctx)
    _echo_if_not_quiet(ctx, manager.format_config(config))


@config_group.command("get")
@click.argument("key")
@handle_errors
def config_get(key):
    """Get a config value."""
    ctx = click.get_current_context()
    manager, _ = _get_config_manager(ctx)
    _echo_if_not_quiet(ctx, manager.get(key))


@config_group.command("set")
@click.argument("key")
@click.argument("value")
@handle_errors
def config_set(key, value):
    """Set a config value."""
    ctx = click.get_current_context()
    manager, config = _get_config_manager(ctx)
    storage = _get_storage(ctx)
    warning = ""
    if key in {"embedding.model", "embedding.provider"} and any(repo.chunk_count > 0 for repo in storage.list_repos()):
        warning = f"Warning: changing {key} may require re-indexing existing data."
    manager.set(key, value)
    manager.save(ctx.obj["config_path"], config)
    if warning and not ctx.obj["quiet"]:
        click.echo(warning)
    if not ctx.obj["quiet"]:
        click.echo(f"Updated {key}")


@cli.command()
@handle_errors
def mcp():
    """Run the MCP server over stdio."""
    ctx = click.get_current_context()
    storage = _get_storage(ctx)
    repos = storage.list_repos()
    if not repos or not any(repo.chunk_count > 0 for repo in repos):
        raise CodeSearchError("Index is empty. Run codesearch index first.")
    from codesearch.mcp_server import CodeSearchMCPServer
    from codesearch.searcher import Searcher

    provider = create_provider(_build_provider_config(ctx))
    searcher = Searcher(storage=storage, provider=provider)
    server = CodeSearchMCPServer(searcher)
    server.run()
