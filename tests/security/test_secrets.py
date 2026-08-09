"""Tests for secret references, resolvers, and ephemeral values."""

from __future__ import annotations

from typing import Any

import pytest

from src.persistence.configuration import ConfigurationError
from src.security.secrets import (
    EnvironmentSecretResolver,
    MissingSecretError,
    SecretReference,
    SecretResolverRegistry,
    SecretSource,
    SecretValue,
    default_resolver_registry,
    resolve_credential,
)

SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_7a8b9c"


def test_secret_reference_serializes_safely() -> None:
    ref = SecretReference(source=SecretSource.ENVIRONMENT, name="OPENAI_API_KEY")
    data = str(ref)
    assert "sk-" not in data
    assert "OPENAI_API_KEY" in data


def test_secret_value_redacts_str_and_repr() -> None:
    value = SecretValue(SENTINEL)
    assert str(value) == "<redacted>"
    assert repr(value) == "SecretValue(<redacted>)"
    assert SENTINEL not in str(value)
    assert SENTINEL not in repr(value)


def test_secret_value_reveals_only_when_explicit() -> None:
    value = SecretValue(SENTINEL)
    assert value.reveal() == SENTINEL


def test_environment_resolver_reads_only_referenced_variable(monkeypatch: Any) -> None:
    monkeypatch.setenv("OMNIFORGE_TEST_OPENAI_KEY", SENTINEL)
    resolver = EnvironmentSecretResolver()
    ref = SecretReference(source=SecretSource.ENVIRONMENT, name="OMNIFORGE_TEST_OPENAI_KEY")
    value = resolver.resolve(ref)
    assert value.reveal() == SENTINEL


def test_environment_resolver_does_not_scrape_environment() -> None:
    resolver = EnvironmentSecretResolver()
    # No attribute or method exposes the full environment mapping.
    assert not hasattr(resolver, "_environment")


def test_environment_resolver_rejects_missing_variable() -> None:
    resolver = EnvironmentSecretResolver()
    ref = SecretReference(source=SecretSource.ENVIRONMENT, name="OMNIFORGE_TEST_MISSING_SECRET")
    with pytest.raises(MissingSecretError) as caught:
        resolver.resolve(ref)
    assert "not set" in str(caught.value)
    assert SENTINEL not in str(caught.value)


def test_environment_resolver_rejects_empty_by_default(monkeypatch: Any) -> None:
    monkeypatch.setenv("OMNIFORGE_TEST_EMPTY_SECRET", "")
    resolver = EnvironmentSecretResolver()
    ref = SecretReference(source=SecretSource.ENVIRONMENT, name="OMNIFORGE_TEST_EMPTY_SECRET")
    with pytest.raises(MissingSecretError) as caught:
        resolver.resolve(ref)
    assert "empty" in str(caught.value)


def test_environment_resolver_allows_empty_when_configured(monkeypatch: Any) -> None:
    monkeypatch.setenv("OMNIFORGE_TEST_EMPTY_SECRET_OK", "")
    resolver = EnvironmentSecretResolver()
    ref = SecretReference(
        source=SecretSource.ENVIRONMENT,
        name="OMNIFORGE_TEST_EMPTY_SECRET_OK",
        metadata={"allow_empty": True},
    )
    value = resolver.resolve(ref)
    assert value.reveal() == ""


def test_registry_routes_by_source() -> None:
    registry = SecretResolverRegistry()
    registry.register(SecretSource.ENVIRONMENT, EnvironmentSecretResolver())
    ref = SecretReference(source=SecretSource.ENVIRONMENT, name="OMNIFORGE_TEST_REGISTRY_KEY")
    with pytest.raises(MissingSecretError):
        registry.resolve(ref)


def test_registry_rejects_unregistered_source() -> None:
    registry = SecretResolverRegistry()
    ref = SecretReference(source=SecretSource.ENVIRONMENT, name="OMNIFORGE_TEST_KEY")
    with pytest.raises(MissingSecretError) as caught:
        registry.resolve(ref)
    assert "no resolver registered" in str(caught.value)


def test_default_registry_includes_environment_resolver(monkeypatch: Any) -> None:
    monkeypatch.setenv("OMNIFORGE_TEST_DEFAULT_REGISTRY", SENTINEL)
    registry = default_resolver_registry()
    ref = SecretReference(source=SecretSource.ENVIRONMENT, name="OMNIFORGE_TEST_DEFAULT_REGISTRY")
    assert registry.resolve(ref).reveal() == SENTINEL


def test_resolve_credential_accepts_reference() -> None:
    ref = SecretReference(source=SecretSource.ENVIRONMENT, name="OMNIFORGE_TEST_REF")
    registry = SecretResolverRegistry()
    registry.register(SecretSource.ENVIRONMENT, _ConstantResolver(SENTINEL))
    value = resolve_credential(ref, registry)
    assert value is not None
    assert value.reveal() == SENTINEL


def test_resolve_credential_accepts_dict() -> None:
    registry = SecretResolverRegistry()
    registry.register(SecretSource.ENVIRONMENT, _ConstantResolver(SENTINEL))
    value = resolve_credential({"source": "environment", "name": "X"}, registry)
    assert value is not None
    assert value.reveal() == SENTINEL


def test_resolve_credential_rejects_raw_string() -> None:
    with pytest.raises(ConfigurationError, match="raw value"):
        resolve_credential(SENTINEL)


def test_resolve_credential_returns_none_for_none() -> None:
    assert resolve_credential(None) is None


def test_secret_reference_rejects_arbitrary_metadata() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        SecretReference(
            source=SecretSource.ENVIRONMENT,
            name="OPENAI_API_KEY",
            metadata={"password": SENTINEL},
        )


def test_secret_reference_metadata_allow_empty_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="boolean"):
        SecretReference(
            source=SecretSource.ENVIRONMENT,
            name="OPENAI_API_KEY",
            metadata={"allow_empty": "yes"},
        )


def test_secret_reference_allows_empty_metadata() -> None:
    ref = SecretReference(source=SecretSource.ENVIRONMENT, name="OPENAI_API_KEY")
    assert ref.metadata == {}


def test_resolve_credential_rejects_unknown_fields() -> None:
    with pytest.raises(ConfigurationError, match="unknown field"):
        resolve_credential({"source": "environment", "name": "X", "password": SENTINEL})


def test_resolve_credential_rejects_metadata_smuggling() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        resolve_credential(
            {
                "source": "environment",
                "name": "X",
                "metadata": {"password": SENTINEL},
            }
        )


class _ConstantResolver:
    def __init__(self, value: str) -> None:
        self._value = value

    def resolve(self, reference: SecretReference) -> SecretValue:
        return SecretValue(self._value)
