import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from codesearch.errors import ConfigError

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

try:
    import tomli_w
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local environments
    tomli_w = None


KNOWN_KEYS = {
    "embedding.provider",
    "embedding.model",
    "embedding.backend",
    "embedding.api_key",
    "embedding.base_url",
}


@dataclass
class EmbeddingConfig:
    provider: str = "onnx"
    model: str = "all-MiniLM-L6-v2"
    backend: str = "onnx"
    api_key: str | None = None
    base_url: str | None = None


@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


class ConfigManager:
    def __init__(self):
        self._config: AppConfig = AppConfig()

    def load(self, path: Path) -> AppConfig:
        path = Path(path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._config = AppConfig()
            self._write_toml(path, self._config)
            return self._config

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Invalid config file: {e}")

        emb_data = data.get("embedding", {})
        emb = EmbeddingConfig(
            provider=emb_data.get("provider", EmbeddingConfig.provider),
            model=emb_data.get("model", EmbeddingConfig.model),
            backend=emb_data.get("backend", EmbeddingConfig.backend),
            api_key=emb_data.get("api_key"),
            base_url=emb_data.get("base_url"),
        )
        self._config = AppConfig(embedding=emb)
        return self._config

    def save(self, path: Path, config: AppConfig) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_toml(path, config)
        self._config = config

    def get(self, key: str) -> str:
        if key not in KNOWN_KEYS:
            raise ConfigError(f"Unknown config key: {key}. Known keys: {', '.join(sorted(KNOWN_KEYS))}")
        section, attr = key.split(".", 1)
        obj = getattr(self._config, section, None)
        if obj is None:
            raise ConfigError(f"Unknown config section: {section}")
        value = getattr(obj, attr, None)
        return str(value) if value is not None else ""

    def set(self, key: str, value: str) -> None:
        if key not in KNOWN_KEYS:
            raise ConfigError(f"Unknown config key: {key}. Known keys: {', '.join(sorted(KNOWN_KEYS))}")
        section, attr = key.split(".", 1)
        obj = getattr(self._config, section, None)
        if obj is None:
            raise ConfigError(f"Unknown config section: {section}")
        setattr(obj, attr, value)

    def resolve_api_key(self, config: EmbeddingConfig) -> str | None:
        return os.environ.get("CODESEARCH_API_KEY") or config.api_key

    def format_config(self, config: AppConfig) -> str:
        lines = []
        lines.append("[embedding]")
        emb = config.embedding
        lines.append(f"  provider = {emb.provider}")
        lines.append(f"  model = {emb.model}")
        lines.append(f"  backend = {emb.backend}")
        if emb.api_key:
            lines.append(f"  api_key = ***")
        if emb.base_url:
            lines.append(f"  base_url = {emb.base_url}")
        return "\n".join(lines)

    def _write_toml(self, path: Path, config: AppConfig) -> None:
        data = {}
        emb = config.embedding
        emb_dict = {}
        emb_dict["provider"] = emb.provider
        emb_dict["model"] = emb.model
        emb_dict["backend"] = emb.backend
        if emb.api_key is not None:
            emb_dict["api_key"] = emb.api_key
        if emb.base_url is not None:
            emb_dict["base_url"] = emb.base_url
        data["embedding"] = emb_dict

        with open(path, "wb") as f:
            if tomli_w is not None:
                tomli_w.dump(data, f)
            else:
                f.write(self._render_toml(data).encode("utf-8"))

        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
        except OSError:
            pass

    def _render_toml(self, data: dict[str, dict[str, str]]) -> str:
        lines: list[str] = []
        for section, values in data.items():
            lines.append(f"[{section}]")
            for key, value in values.items():
                lines.append(f"{key} = {json.dumps(value)}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
