"""Provider-neutral secret references and resolvers.

Secret values must never be persisted in normal configuration or runtime state.
This module provides the reference model, an ephemeral value wrapper, and an
explicit environment-variable resolver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from src.persistence.configuration import ConfigurationError


class SecretSource(StrEnum):
    """Supported secret sources."""

    ENVIRONMENT = "environment"


def _validate_secret_reference_metadata(metadata: dict[str, Any]) -> None:
    """Ensure metadata only contains safe, known keys.

    Arbitrary metadata must not become a secret-smuggling channel.
    """
    for key, value in metadata.items():
        if key != "allow_empty":
            raise ValueError(
                f"SecretReference metadata contains unknown key {key!r}; "
                "only 'allow_empty' is permitted"
            )
        if not isinstance(value, bool):
            raise ValueError("SecretReference metadata 'allow_empty' must be a boolean")


@dataclass(frozen=True)
class SecretReference:
    """A safe, persistable reference to a secret. No value is stored here."""

    source: SecretSource
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("SecretReference.name must be non-empty")
        if self.source == SecretSource.ENVIRONMENT and "=" in self.name:
            raise ValueError("environment secret name must not contain '='")
        if not isinstance(self.metadata, dict):
            raise ValueError("SecretReference.metadata must be a dict")
        _validate_secret_reference_metadata(self.metadata)


@dataclass(frozen=True)
class SecretValue:
    """Ephemeral wrapper around a resolved secret.

    ``str()`` and ``repr()`` are redacted to prevent accidental logging.
    Adapters consume secrets through ``reveal()``.
    """

    _value: str

    def __str__(self) -> str:
        return "<redacted>"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    def reveal(self) -> str:
        """Return the raw secret value for immediate adapter consumption."""
        return self._value


class MissingSecretError(ConfigurationError):
    """Raised when a referenced secret cannot be resolved."""

    def __init__(self, reference: SecretReference, reason: str | None = None) -> None:
        self.reference = reference
        message = f"Missing secret {reference.source.value}/{reference.name}"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message)


@runtime_checkable
class SecretResolver(Protocol):
    """Protocol for secret resolvers. Implementations must be explicit and mockable."""

    def resolve(self, reference: SecretReference) -> SecretValue:
        """Resolve a reference to an ephemeral SecretValue."""
        ...


class EnvironmentSecretResolver:
    """Resolves secrets from explicitly named environment variables.

    No global environment scraping occurs. Empty values are treated as missing
    unless the reference metadata contains ``allow_empty: true``.
    """

    def resolve(self, reference: SecretReference) -> SecretValue:
        if reference.source is not SecretSource.ENVIRONMENT:
            raise MissingSecretError(
                reference,
                reason=f"{type(self).__name__} cannot resolve source {reference.source.value}",
            )

        value = os.environ.get(reference.name)
        allow_empty = reference.metadata.get("allow_empty", False)

        if value is None:
            raise MissingSecretError(reference, reason="environment variable not set")
        if not allow_empty and value == "":
            raise MissingSecretError(reference, reason="environment variable is empty")

        return SecretValue(value)


class SecretResolverRegistry:
    """Registry for secret resolvers by source."""

    def __init__(self) -> None:
        self._resolvers: dict[SecretSource, SecretResolver] = {}

    def register(self, source: SecretSource, resolver: SecretResolver) -> None:
        self._resolvers[source] = resolver

    def resolve(self, reference: SecretReference) -> SecretValue:
        resolver = self._resolvers.get(reference.source)
        if resolver is None:
            raise MissingSecretError(
                reference, reason=f"no resolver registered for source {reference.source.value}"
            )
        return resolver.resolve(reference)


def default_resolver_registry() -> SecretResolverRegistry:
    """Return a registry with the standard environment resolver."""
    registry = SecretResolverRegistry()
    registry.register(SecretSource.ENVIRONMENT, EnvironmentSecretResolver())
    return registry


_CREDENTIAL_REF_ALLOWED_FIELDS = frozenset({"source", "name", "reference_id", "metadata"})


def resolve_credential(
    credential_ref: SecretReference | dict[str, Any] | str | None,
    resolver_registry: SecretResolverRegistry | None = None,
) -> SecretValue | None:
    """Resolve a credential reference into an ephemeral SecretValue.

    Accepts a ``SecretReference``, a dict shaped like one, or ``None``.
    Returns ``None`` when no reference is supplied.
    """
    if credential_ref is None:
        return None

    if isinstance(credential_ref, SecretReference):
        reference = credential_ref
    elif isinstance(credential_ref, dict):
        for key in credential_ref:
            if key not in _CREDENTIAL_REF_ALLOWED_FIELDS:
                raise ConfigurationError(f"credential_ref contains unknown field {key!r}")
        source = SecretSource(credential_ref.get("source", SecretSource.ENVIRONMENT.value))
        name = credential_ref.get("name") or credential_ref.get("reference_id", "")
        metadata = dict(credential_ref.get("metadata", {}))
        reference = SecretReference(source=source, name=name, metadata=metadata)
    else:
        # Legacy or unsafe raw value path: reject.
        raise ConfigurationError(
            "credential_ref must be a SecretReference or dict, not a raw value"
        )

    registry = resolver_registry or default_resolver_registry()
    return registry.resolve(reference)
