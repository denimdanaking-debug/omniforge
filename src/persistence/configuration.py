"""Versioned OmniForge configuration loading and migration hooks."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.routing.policy import ProjectRoutingPolicy, RoutingPin
from src.security.redaction import SENSITIVE_STRUCTURED_KEYS, _normalize_key

CURRENT_CONFIG_SCHEMA_VERSION = "1.4.0"
SUPPORTED_CONFIG_SCHEMA_VERSIONS = frozenset({CURRENT_CONFIG_SCHEMA_VERSION})

Migration = Callable[[dict[str, Any]], dict[str, Any]]
_MIGRATIONS: dict[str, tuple[str, Migration]] = {}


class ConfigurationError(ValueError):
    """Base class for safe configuration failures."""


class InvalidConfiguration(ConfigurationError):
    """Raised when configuration content is malformed or incomplete."""


class UnsupportedConfigVersion(ConfigurationError):
    """Raised when a configuration version cannot be consumed safely."""


class RawSecretInConfigurationError(InvalidConfiguration):
    """Raised when raw secret material is detected in normal configuration."""


class SecretValidationError(InvalidConfiguration):
    """Raised when a secret reference is malformed."""


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


@register_migration("1.0.0", "1.1.0")
def _migrate_1_0_0_to_1_1_0(old: dict[str, Any]) -> dict[str, Any]:
    """Add Phase 5 administrative fields with safe defaults."""
    working = copy.deepcopy(old)
    providers = working.setdefault("providers", {})
    for provider in providers.values():
        if isinstance(provider, dict):
            provider.setdefault("models", {})
            provider.setdefault("routes", {})
    working.setdefault("pins", {})
    working.setdefault("project_policies", {})
    return working


@register_migration("1.1.0", "1.2.0")
def _migrate_1_1_0_to_1_2_0(old: dict[str, Any]) -> dict[str, Any]:
    """Add Phase 8 dynamic router configuration with safe defaults."""
    working = copy.deepcopy(old)
    working.setdefault("routing_mode", "legacy")
    working.setdefault("exploration_enabled", False)
    router_config = working.setdefault("router_config", {})
    router_config.setdefault("factor_weights", {})
    router_config.setdefault("priors", [])
    router_config.setdefault("emergency_fallback_orders", {})
    router_config.setdefault("cost_metadata", {})
    router_config.setdefault("default_safety_margin_fraction", 0.1)
    router_config.setdefault("exploration_enabled", False)
    return working


@register_migration("1.2.0", "1.3.0")
def _migrate_1_2_0_to_1_3_0(old: dict[str, Any]) -> dict[str, Any]:
    """Add Phase 9 risk policy with safe defaults; preserve all prior settings."""
    working = copy.deepcopy(old)
    working.setdefault("routing_mode", "legacy")
    working.setdefault("exploration_enabled", False)
    working.setdefault("dynamic_routing_enabled", False)
    working.setdefault("risk_policy", {})
    project_policies = working.setdefault("project_policies", {})
    for policy in project_policies.values():
        if isinstance(policy, dict):
            policy.setdefault("risk_policy", {})
    router_config = working.setdefault("router_config", {})
    router_config.setdefault("exploration_enabled", False)
    return working


@register_migration("1.3.0", "1.4.0")
def _migrate_1_3_0_to_1_4_0(old: dict[str, Any]) -> dict[str, Any]:
    """Add Phase 10 failure recovery policy with safe defaults."""
    from src.recovery.retry_policy import FailureRecoveryPolicy

    working = copy.deepcopy(old)
    working.setdefault("routing_mode", "legacy")
    working.setdefault("exploration_enabled", False)
    working.setdefault("dynamic_routing_enabled", False)
    working.setdefault("recovery_policy", FailureRecoveryPolicy().to_dict())
    router_config = working.setdefault("router_config", {})
    router_config.setdefault("exploration_enabled", False)
    return working


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


_SECRET_REFERENCE_ALLOWED_FIELDS = frozenset({"source", "name", "reference_id", "metadata"})


def _is_secret_reference(value: Any) -> bool:
    """Return True if ``value`` is a dict shaped exactly like a SecretReference."""
    if not isinstance(value, dict):
        return False
    if not set(value.keys()).issubset(_SECRET_REFERENCE_ALLOWED_FIELDS):
        return False
    if "source" not in value:
        return False
    return "name" in value or "reference_id" in value


def _is_sensitive_structured_key(key: Any) -> bool:
    """Return True if ``key`` matches a canonical secret-bearing field name."""
    return isinstance(key, str) and _normalize_key(key) in SENSITIVE_STRUCTURED_KEYS


def _validate_no_raw_secrets(path: str, obj: Any) -> None:
    """Recursively reject raw secret fields outside of SecretReference values."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                continue
            if _is_sensitive_structured_key(key) and not _is_secret_reference(value):
                raise RawSecretInConfigurationError(
                    f"raw secret field {key!r} at {path} is not allowed in normal configuration; "
                    "use a SecretReference via credential_ref"
                )
            _validate_no_raw_secrets(f"{path}.{key}", value)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            _validate_no_raw_secrets(f"{path}[{index}]", item)


def _validate_secret_reference(path: str, value: Any) -> None:
    """Validate a credential_ref value against the SecretReference schema."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SecretValidationError(f"credential_ref at {path} must be a SecretReference object")
    for key in value:
        if key not in _SECRET_REFERENCE_ALLOWED_FIELDS:
            raise SecretValidationError(f"credential_ref at {path} contains unknown field {key!r}")
    source = value.get("source")
    if source not in {"environment"}:
        raise SecretValidationError(f"credential_ref at {path} has unsupported source {source!r}")
    name = value.get("name") or value.get("reference_id")
    if not isinstance(name, str) or not name.strip():
        raise SecretValidationError(
            f"credential_ref at {path} requires a non-empty name/reference_id"
        )
    metadata = value.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise SecretValidationError(f"credential_ref metadata at {path} must be an object")
        for meta_key, meta_value in metadata.items():
            if meta_key != "allow_empty":
                raise SecretValidationError(
                    f"credential_ref metadata at {path} contains unknown key {meta_key!r}"
                )
            if not isinstance(meta_value, bool):
                raise SecretValidationError(
                    f"credential_ref metadata.allow_empty at {path} must be a boolean"
                )


def _validate_pin(path: str, value: Any) -> RoutingPin:
    if not isinstance(value, dict):
        raise InvalidConfiguration(f"pin at {path} must be an object")
    allowed = {"provider_id", "model_id", "route_id"}
    for key in value:
        if key not in allowed:
            raise InvalidConfiguration(f"pin at {path} contains unknown field {key!r}")
    for key in allowed:
        field_value = value.get(key)
        if field_value is not None and (not isinstance(field_value, str) or not field_value):
            raise InvalidConfiguration(
                f"pin field {key} at {path} must be a non-empty string or absent"
            )
    return RoutingPin(
        provider_id=value.get("provider_id"),
        model_id=value.get("model_id"),
        route_id=value.get("route_id"),
    )


def _validate_project_policy(path: str, value: Any) -> ProjectRoutingPolicy:
    from src.risk.project_policy import ProjectRiskPolicy

    if not isinstance(value, dict):
        raise InvalidConfiguration(f"project policy at {path} must be an object")
    try:
        _ = ProjectRoutingPolicy.from_dict(value)
    except ValueError as exc:
        raise InvalidConfiguration(f"project policy at {path} is invalid: {exc}") from exc

    risk_policy_raw = value.get("risk_policy", {})
    try:
        risk_policy = ProjectRiskPolicy.from_dict(risk_policy_raw)
    except ValueError as exc:
        raise InvalidConfiguration(f"{path}.risk_policy is invalid: {exc}") from exc

    value["risk_policy"] = risk_policy.to_dict()
    # Re-create the routing policy from the normalized dict so that its own
    # serialized form reflects the validated risk policy.
    return ProjectRoutingPolicy.from_dict(value)


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

    _validate_no_raw_secrets("config", working)

    for provider_id, provider in providers.items():
        if not isinstance(provider_id, str) or not provider_id:
            raise InvalidConfiguration("provider identifiers must be non-empty strings")
        if not isinstance(provider, dict):
            raise InvalidConfiguration(f"provider {provider_id!r} must be an object")
        if not isinstance(provider.get("enabled"), bool):
            raise InvalidConfiguration(f"provider {provider_id!r} requires boolean enabled")

        _validate_secret_reference(
            f"providers.{provider_id}.credential_ref", provider.get("credential_ref")
        )

        models = provider.get("models", {})
        if not isinstance(models, dict):
            raise InvalidConfiguration(f"provider {provider_id!r} models must be an object")
        for model_id, model in models.items():
            if not isinstance(model_id, str) or not model_id:
                raise InvalidConfiguration(
                    f"provider {provider_id!r} model ids must be non-empty strings"
                )
            if not isinstance(model, dict):
                raise InvalidConfiguration(
                    f"provider {provider_id!r} model {model_id!r} must be an object"
                )
            if not isinstance(model.get("enabled", True), bool):
                raise InvalidConfiguration(
                    f"provider {provider_id!r} model {model_id!r} enabled must be a boolean"
                )

        routes = provider.get("routes", {})
        if not isinstance(routes, dict):
            raise InvalidConfiguration(f"provider {provider_id!r} routes must be an object")
        for route_id, route in routes.items():
            if not isinstance(route_id, str) or not route_id:
                raise InvalidConfiguration(
                    f"provider {provider_id!r} route ids must be non-empty strings"
                )
            if not isinstance(route, dict):
                raise InvalidConfiguration(
                    f"provider {provider_id!r} route {route_id!r} must be an object"
                )
            if not isinstance(route.get("enabled", True), bool):
                raise InvalidConfiguration(
                    f"provider {provider_id!r} route {route_id!r} enabled must be a boolean"
                )

    pins = working.get("pins", {})
    if not isinstance(pins, dict):
        raise InvalidConfiguration("pins must be an object")
    for pin_id, pin in pins.items():
        if not isinstance(pin_id, str) or not pin_id:
            raise InvalidConfiguration("pin identifiers must be non-empty strings")
        _validate_pin(f"pins.{pin_id}", pin)

    project_policies = working.get("project_policies", {})
    if not isinstance(project_policies, dict):
        raise InvalidConfiguration("project_policies must be an object")
    validated_policies: dict[str, dict[str, Any]] = {}
    for project_id, policy in project_policies.items():
        if not isinstance(project_id, str) or not project_id:
            raise InvalidConfiguration("project policy identifiers must be non-empty strings")
        validated_policy = _validate_project_policy(f"project_policies.{project_id}", policy)
        validated_policies[project_id] = validated_policy.to_dict()
    working["project_policies"] = validated_policies

    router_config = working.get("router_config", {})
    _validate_router_config("router_config", router_config)

    risk_policy = working.get("risk_policy", {})
    _validate_risk_policy("risk_policy", risk_policy)

    recovery_policy = working.get("recovery_policy", {})
    _validate_recovery_policy("recovery_policy", recovery_policy)

    return working


def _validate_router_config(path: str, value: Any) -> None:
    from src.routing.dynamic.config import RouterConfig, RouterConfigError

    if not isinstance(value, dict):
        raise InvalidConfiguration(f"{path} must be an object")
    try:
        RouterConfig.from_dict(value)
    except (RouterConfigError, ValueError) as exc:
        raise InvalidConfiguration(f"{path} is invalid: {exc}") from exc


def _validate_risk_policy(path: str, value: Any) -> None:
    from src.risk.project_policy import ProjectRiskPolicy

    if not isinstance(value, dict):
        raise InvalidConfiguration(f"{path} must be an object")
    try:
        ProjectRiskPolicy.from_dict(value)
    except ValueError as exc:
        raise InvalidConfiguration(f"{path} is invalid: {exc}") from exc


def _validate_recovery_policy(path: str, value: Any) -> None:
    from src.recovery.retry_policy import FailureRecoveryPolicy

    if not isinstance(value, dict):
        raise InvalidConfiguration(f"{path} must be an object")
    try:
        FailureRecoveryPolicy.from_dict(value)
    except ValueError as exc:
        raise InvalidConfiguration(f"{path} is invalid: {exc}") from exc


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


def extract_administrative_state(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a runtime-state administrative snapshot from validated config.

    Resolved secrets are never included.
    """
    working = validate_config(config)
    provider_status: dict[str, dict[str, Any]] = {}
    model_status: dict[str, dict[str, Any]] = {}
    route_status: dict[str, dict[str, Any]] = {}

    for provider_id, provider in working.get("providers", {}).items():
        provider_status[provider_id] = {"enabled": provider.get("enabled", True)}
        for model_id, model in provider.get("models", {}).items():
            model_status[model_id] = {"enabled": model.get("enabled", True)}
        for route_id, route in provider.get("routes", {}).items():
            route_status[route_id] = {"enabled": route.get("enabled", True)}

    return {
        "schema_version": CURRENT_CONFIG_SCHEMA_VERSION,
        "provider_status": provider_status,
        "model_status": model_status,
        "route_status": route_status,
        "routing_mode": working.get("routing_mode", "legacy"),
        "dynamic_routing_enabled": working.get("dynamic_routing_enabled", False),
        "exploration_enabled": working.get("exploration_enabled", False),
        "pins": working.get("pins", {}),
        "project_policies": copy.deepcopy(working.get("project_policies", {})),
        "router_config": copy.deepcopy(working.get("router_config", {})),
        "recovery_policy": copy.deepcopy(working.get("recovery_policy", {})),
    }


def extract_router_config(admin_state: dict[str, Any]) -> Any:
    """Build a validated RouterConfig from administrative state."""
    from src.routing.dynamic.config import RouterConfigError, load_router_config

    raw = admin_state.get("router_config", {})
    if not isinstance(raw, dict):
        raise InvalidConfiguration("router_config must be an object")
    try:
        config = load_router_config(raw)
    except RouterConfigError as exc:
        raise InvalidConfiguration(f"router_config is invalid: {exc}") from exc
    return config
