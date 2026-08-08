"""Versioned OmniForge configuration loading and migration hooks."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

CURRENT_CONFIG_SCHEMA_VERSION = "1.0.0"
SUPPORTED_CONFIG_SCHEMA_VERSIONS = frozenset({CURRENT_CONFIG_SCHEMA_VERSION})

Migration = Callable[[dict[str, Any]], dict[str, Any]]
_MIGRATIONS: dict[str, tuple[str, Migration]] = {}


class ConfigurationError(ValueError):
    """Base class for safe configuration failures."""


class InvalidConfiguration(ConfigurationError):
    """Raised when configuration content is malformed or incomplete."""


class UnsupportedConfigVersion(ConfigurationError):
    """Raised when a configuration version cannot be consumed safely."""


def register_migration(from_version: str, to_version: str) -> Callable[[Migration], Migration]:
    """Register one explicit, deterministic schema migration edge."""

    if not from_version or not to_version or from_version == to_version:
        raise ValueError("migration versions must be distinct non-empty strings")

    def decorator(function: Migration) -> Migration:
        if from_version in _MIGRATIONS:
            raise ValueError(f"migration from {from_version} is already registered")
        _MIGRATIONS[from_version] = (to_version, function)
        return function

    return decorator


def _require_version(config: Mapping[str, Any]) -> str:
    version = config.get("schema_version")
    if not isinstance(version, str) or not version.strip():
        raise InvalidConfiguration("configuration requires a non-empty string schema_version")
    return version


def migrate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate a configuration to the current version using registered hooks.

    Unknown versions fail closed. The input mapping is never modified.
    """

    working = copy.deepcopy(dict(config))
    version = _require_version(working)
    visited: set[str] = set()

    while version != CURRENT_CONFIG_SCHEMA_VERSION:
        if version in visited:
            raise UnsupportedConfigVersion(f"configuration migration cycle detected at {version}")
        visited.add(version)

        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise UnsupportedConfigVersion(
                f"unsupported configuration schema_version {version!r}; "
                f"current version is {CURRENT_CONFIG_SCHEMA_VERSION!r}"
            )

        to_version, migrate = migration
        migrated = migrate(copy.deepcopy(working))
        if not isinstance(migrated, dict):
            raise InvalidConfiguration(
                f"migration {version}->{to_version} did not return an object"
            )
        migrated["schema_version"] = to_version
        working = migrated
        version = to_version

    return working


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the current bootstrap configuration contract."""

    working = migrate_config(config)
    version = _require_version(working)
    if version not in SUPPORTED_CONFIG_SCHEMA_VERSIONS:
        raise UnsupportedConfigVersion(f"configuration schema_version {version!r} is not supported")

    routing_mode = working.get("routing_mode")
    if routing_mode not in {"legacy", "dynamic"}:
        raise InvalidConfiguration("routing_mode must be 'legacy' or 'dynamic'")

    providers = working.get("providers")
    if not isinstance(providers, dict):
        raise InvalidConfiguration("providers must be an object")

    exploration = working.get("exploration_enabled", False)
    if not isinstance(exploration, bool):
        raise InvalidConfiguration("exploration_enabled must be a boolean")

    for provider_id, provider in providers.items():
        if not isinstance(provider_id, str) or not provider_id:
            raise InvalidConfiguration("provider identifiers must be non-empty strings")
        if not isinstance(provider, dict):
            raise InvalidConfiguration(f"provider {provider_id!r} must be an object")
        if not isinstance(provider.get("enabled"), bool):
            raise InvalidConfiguration(f"provider {provider_id!r} requires boolean enabled")

    return working


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, migrate, and validate a JSON configuration file."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InvalidConfiguration(f"unable to read configuration {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InvalidConfiguration(f"configuration {source} is invalid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise InvalidConfiguration("configuration root must be an object")
    return validate_config(raw)
