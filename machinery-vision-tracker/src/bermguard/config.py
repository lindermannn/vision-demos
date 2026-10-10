"""Configuration loading with a reproducible raw-content hash."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class LoadedConfiguration:
    data: Mapping[str, Any]
    sha256: str
    source: str

    @property
    def fail_fast(self) -> bool:
        runtime = self.data.get("runtime", {})
        return bool(runtime.get("fail_fast", False))


def _validate(data: Any) -> Mapping[str, Any]:
    if not isinstance(data, dict):
        raise ConfigurationError("configuration root must be a mapping")
    if data.get("schema_version") != 1:
        raise ConfigurationError("configuration schema_version must equal 1")
    runtime = data.get("runtime")
    if not isinstance(runtime, dict):
        raise ConfigurationError("configuration runtime section must be a mapping")
    if not isinstance(runtime.get("fail_fast"), bool):
        raise ConfigurationError("runtime.fail_fast must be a boolean")
    output = data.get("output")
    if not isinstance(output, dict):
        raise ConfigurationError("configuration output section must be a mapping")
    if output.get("overwrite") is not False:
        raise ConfigurationError("output.overwrite must remain false")
    return data


def load_configuration(path: Optional[Path]) -> LoadedConfiguration:
    if path is None:
        resource = resources.files("bermguard").joinpath("default_config.yaml")
        raw = resource.read_bytes()
        source = "built-in:default_config.yaml"
    else:
        try:
            resolved = path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ConfigurationError(f"configuration file not found: {path.name}") from exc
        if not resolved.is_file():
            raise ConfigurationError(f"configuration path is not a file: {path.name}")
        raw = resolved.read_bytes()
        source = resolved.name

    try:
        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationError("configuration must be valid UTF-8 YAML") from exc

    return LoadedConfiguration(
        data=_validate(parsed),
        sha256=hashlib.sha256(raw).hexdigest(),
        source=source,
    )
