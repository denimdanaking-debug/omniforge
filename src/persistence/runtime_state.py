"""Versioned, restart-safe OmniForge runtime-state persistence."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.security.redaction import redact
from src.security.secrets import SecretValue

CURRENT_RUNTIME_STATE_VERSION = "1.1.0"
RuntimeMigration = Callable[[dict[str, Any]], dict[str, Any]]
_RUNTIME_MIGRATIONS: dict[str, tuple[str, RuntimeMigration]] = {}

ALLOWED_WORKFLOW_STATES = frozenset(
    {
        "STOPPED",
        "PLANNING",
        "EXECUTING",
        "WAITING_FOR_PROVIDER",
        "WAITING_FOR_RETRY",
        "VALIDATING",
        "REVIEWING",
        "ARBITRATING",
        "REPAIRING",
        "INTEGRATING",
        "BLOCKED",
        "COMPLETE",
    }
)

ALLOWED_ROUTING_MODES = frozenset({"legacy", "dynamic"})


@dataclass(frozen=True)
class RuntimeStateDiagnostic:
    code: str
    message: str
    path: str | None = None
    recoverable: bool = True


class RuntimeStateError(RuntimeError):
    def __init__(self, diagnostic: RuntimeStateDiagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class CorruptRuntimeState(RuntimeStateError):
    pass


class UnsupportedRuntimeStateVersion(RuntimeStateError):
    pass


def register_runtime_migration(
    from_version: str, to_version: str
) -> Callable[[RuntimeMigration], RuntimeMigration]:
    if not from_version or not to_version or from_version == to_version:
        raise ValueError("runtime migration versions must be distinct non-empty strings")

    def decorator(function: RuntimeMigration) -> RuntimeMigration:
        if from_version in _RUNTIME_MIGRATIONS:
            raise ValueError(f"runtime migration from {from_version} is already registered")
        _RUNTIME_MIGRATIONS[from_version] = (to_version, function)
        return function

    return decorator


@register_runtime_migration("1.0.0", "1.1.0")
def _migrate_runtime_1_0_0_to_1_1_0(old: dict[str, Any]) -> dict[str, Any]:
    """Add Phase 5 administrative state fields with safe defaults."""
    working = copy.deepcopy(old)
    working.setdefault("provider_status", {})
    working.setdefault("model_status", {})
    working.setdefault("route_status", {})
    working.setdefault("routing_mode", "legacy")
    working.setdefault("exploration_enabled", False)
    working.setdefault("pins", {})
    working.setdefault("project_policies", {})
    return working


def _require_version(state: Mapping[str, Any]) -> str:
    version = state.get("schema_version")
    if not isinstance(version, str) or not version.strip():
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic(
                "MISSING_SCHEMA_VERSION", "runtime state requires schema_version"
            )
        )
    return version


def migrate_runtime_state(state: Mapping[str, Any]) -> dict[str, Any]:
    working = copy.deepcopy(dict(state))
    version = _require_version(working)
    visited: set[str] = set()

    while version != CURRENT_RUNTIME_STATE_VERSION:
        if version in visited:
            raise UnsupportedRuntimeStateVersion(
                RuntimeStateDiagnostic(
                    "MIGRATION_CYCLE",
                    f"runtime-state migration cycle detected at {version}",
                )
            )
        visited.add(version)
        migration = _RUNTIME_MIGRATIONS.get(version)
        if migration is None:
            raise UnsupportedRuntimeStateVersion(
                RuntimeStateDiagnostic(
                    "UNSUPPORTED_SCHEMA_VERSION",
                    f"unsupported runtime schema_version {version!r}; current version is {CURRENT_RUNTIME_STATE_VERSION!r}",
                )
            )
        to_version, function = migration
        migrated = function(copy.deepcopy(working))
        if not isinstance(migrated, dict):
            raise CorruptRuntimeState(
                RuntimeStateDiagnostic(
                    "INVALID_MIGRATION_RESULT",
                    f"runtime migration {version}->{to_version} did not return an object",
                )
            )
        migrated["schema_version"] = to_version
        working = migrated
        version = to_version

    return working


def _validate_status_map(path: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic("INVALID_STATUS_MAP", f"{path} must be an object")
        )
    for key, entry in value.items():
        if not isinstance(key, str) or not key:
            raise CorruptRuntimeState(
                RuntimeStateDiagnostic(
                    "INVALID_STATUS_KEY", f"{path} keys must be non-empty strings"
                )
            )
        if not isinstance(entry, dict):
            raise CorruptRuntimeState(
                RuntimeStateDiagnostic("INVALID_STATUS_ENTRY", f"{path}.{key} must be an object")
            )
        enabled = entry.get("enabled")
        if not isinstance(enabled, bool):
            raise CorruptRuntimeState(
                RuntimeStateDiagnostic(
                    "INVALID_STATUS_ENABLED", f"{path}.{key}.enabled must be a boolean"
                )
            )


def _validate_pins(value: Any) -> None:
    if not isinstance(value, dict):
        raise CorruptRuntimeState(RuntimeStateDiagnostic("INVALID_PINS", "pins must be an object"))
    for pin_id, pin in value.items():
        if not isinstance(pin_id, str) or not pin_id:
            raise CorruptRuntimeState(
                RuntimeStateDiagnostic("INVALID_PIN_ID", "pin ids must be non-empty strings")
            )
        if not isinstance(pin, dict):
            raise CorruptRuntimeState(
                RuntimeStateDiagnostic("INVALID_PIN", f"pin {pin_id!r} must be an object")
            )
        allowed = {"provider_id", "model_id", "route_id"}
        for key in pin:
            if key not in allowed:
                raise CorruptRuntimeState(
                    RuntimeStateDiagnostic(
                        "UNKNOWN_PIN_FIELD", f"pin {pin_id!r} has unknown field {key!r}"
                    )
                )


def _validate_project_policies(value: Any) -> None:
    if not isinstance(value, dict):
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic("INVALID_PROJECT_POLICIES", "project_policies must be an object")
        )


def validate_runtime_state(state: Mapping[str, Any]) -> dict[str, Any]:
    working = migrate_runtime_state(state)

    run_id = working.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic("INVALID_RUN_ID", "runtime state requires run_id")
        )

    workflow_state = working.get("workflow_state")
    if workflow_state not in ALLOWED_WORKFLOW_STATES:
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic(
                "INVALID_WORKFLOW_STATE",
                f"invalid workflow_state {workflow_state!r}",
            )
        )

    if not isinstance(working.get("checkpoint"), dict):
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic("INVALID_CHECKPOINT", "runtime checkpoint must be an object")
        )

    routing_mode = working.get("routing_mode", "legacy")
    if routing_mode not in ALLOWED_ROUTING_MODES:
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic(
                "INVALID_ROUTING_MODE",
                f"routing_mode must be 'legacy' or 'dynamic', got {routing_mode!r}",
            )
        )

    exploration = working.get("exploration_enabled", False)
    if not isinstance(exploration, bool):
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic("INVALID_EXPLORATION", "exploration_enabled must be a boolean")
        )

    _validate_status_map("provider_status", working.get("provider_status", {}))
    _validate_status_map("model_status", working.get("model_status", {}))
    _validate_status_map("route_status", working.get("route_status", {}))
    _validate_pins(working.get("pins", {}))
    _validate_project_policies(working.get("project_policies", {}))

    return working


def load_runtime_state(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeStateError(
            RuntimeStateDiagnostic("STATE_READ_FAILED", str(exc), str(source))
        ) from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic(
                "CORRUPT_JSON",
                f"runtime state is invalid JSON: {exc}",
                str(source),
            )
        ) from exc

    if not isinstance(raw, dict):
        raise CorruptRuntimeState(
            RuntimeStateDiagnostic(
                "INVALID_ROOT", "runtime state root must be an object", str(source)
            )
        )

    try:
        return validate_runtime_state(raw)
    except RuntimeStateError as exc:
        diagnostic = exc.diagnostic
        if diagnostic.path is None:
            diagnostic = RuntimeStateDiagnostic(
                diagnostic.code, diagnostic.message, str(source), diagnostic.recoverable
            )
        raise type(exc)(diagnostic) from exc


def _json_default(value: Any) -> Any:
    """Fallback JSON encoder that redacts secret wrappers instead of leaking them."""
    if isinstance(value, SecretValue):
        return "<redacted>"
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_runtime_state(path: str | Path, state: Mapping[str, Any]) -> None:
    """Validate and atomically persist state so restart observes all-or-nothing data."""

    destination = Path(path)
    validated = validate_runtime_state(state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Defense-in-depth redaction: ensure no secret-bearing string or structured
    # field can survive into persisted runtime state.
    safe_state = redact(validated)
    payload = json.dumps(safe_state, indent=2, sort_keys=True, default=_json_default) + "\n"

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise
