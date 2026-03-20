from pathlib import Path
import stat

import pytest

from codesearch.config import AppConfig, ConfigManager, EmbeddingConfig
from codesearch.errors import ConfigError


# Test: ConfigManager creates default config on first load
# Validates: REQ-737 (default config file is materialized with default values)
def test_load_creates_default_config_file(tmp_path: Path) -> None:
    path = tmp_path / "codesearch" / "config.toml"

    manager = ConfigManager()
    config = manager.load(path)

    assert path.exists()
    assert config.embedding.provider == "sentence-transformers"
    assert config.embedding.model == "all-MiniLM-L6-v2"
    assert config.embedding.backend == "onnx"


# Test: ConfigManager persists and reloads dot-notation values
# Validates: AC-1360 (REQ-737 - configuration values are saved and loaded consistently)
def test_save_and_reload_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    manager = ConfigManager()
    config = AppConfig(
        embedding=EmbeddingConfig(
            provider="ollama",
            model="qwen3-embedding:0.6b",
            backend="onnx",
            api_key="secret",
            base_url="http://localhost:11434",
        )
    )

    manager.save(path, config)

    reloaded = ConfigManager().load(path)

    assert reloaded.embedding.provider == "ollama"
    assert reloaded.embedding.model == "qwen3-embedding:0.6b"
    assert reloaded.embedding.api_key == "secret"
    assert reloaded.embedding.base_url == "http://localhost:11434"


# Test: ConfigManager get/set works with known dot-notation keys
# Validates: AC-1361 (REQ-737 - config values are addressable via dot-notation)
def test_get_and_set_known_keys() -> None:
    manager = ConfigManager()

    manager.set("embedding.provider", "openai")

    assert manager.get("embedding.provider") == "openai"


# Test: ConfigManager rejects unknown keys
# Validates: AC-1362 (REQ-738 - unknown config keys are rejected)
def test_unknown_key_raises_config_error() -> None:
    manager = ConfigManager()

    with pytest.raises(ConfigError, match="Unknown config key: unknown.key"):
        manager.get("unknown.key")

    with pytest.raises(ConfigError, match="Unknown config key: unknown.key"):
        manager.set("unknown.key", "value")


def test_resolve_api_key_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ConfigManager()
    config = EmbeddingConfig(api_key="from-config")

    monkeypatch.setenv("CODESEARCH_API_KEY", "from-env")

    assert manager.resolve_api_key(config) == "from-env"


def test_save_sets_private_file_permissions(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    manager = ConfigManager()

    manager.save(path, AppConfig())

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
