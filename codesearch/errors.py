class CodeSearchError(Exception):
    """Base class for all codesearch errors."""


class StorageError(CodeSearchError):
    """DB init failures, corruption, locked DB, sqlite-vec extension not found."""


class DimensionMismatchError(StorageError):
    """Index dimensions != current model dimensions. Actionable: run index --full."""


class ProviderError(CodeSearchError):
    """Embedding provider failures: connection, auth, timeout, malformed response."""


class ConfigError(CodeSearchError):
    """Invalid TOML, unknown config key, missing required values."""


class ChunkerError(CodeSearchError):
    """Non-fatal: tree-sitter crash, encoding errors. Logged, file skipped."""
