"""First-class provider identity and operational state."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping

_PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


class ProviderIdentityError(ValueError):
    pass


class ProviderHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    COOLING = "cooling"
    UNAVAILABLE = "unavailable"
    AUTH_FAILED = "auth_failed"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ProviderIdentity:
    """Stable vendor/provider identity, independent of models and routes."""

    provider_id: str
    display_name: str
    failure_domain: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _PROVIDER_ID_PATTERN.fullmatch(self.provider_id):
            raise ProviderIdentityError(
                "provider_id must be 2-64 lowercase letters/digits plus '.', '_' or '-'"
            )
        if not self.display_name.strip():
            raise ProviderIdentityError("display_name must be non-empty")
        if not self.failure_domain.strip():
            raise ProviderIdentityError("failure_domain must be non-empty")


@dataclass(frozen=True)
class ProviderQuotaState:
    remaining_fraction: float | None = None
    reset_at: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.remaining_fraction is not None and not 0.0 <= self.remaining_fraction <= 1.0:
            raise ProviderIdentityError("remaining_fraction must be between 0 and 1")


@dataclass(frozen=True)
class ProviderOperationalState:
    health: ProviderHealth = ProviderHealth.HEALTHY
    quota: ProviderQuotaState = field(default_factory=ProviderQuotaState)
    last_checked_at: str | None = None
    next_retry_at: str | None = None


@dataclass(frozen=True)
class ProviderRegistration:
    identity: ProviderIdentity
    model_ids: frozenset[str] = frozenset()
    route_ids: frozenset[str] = frozenset()


class ProviderRegistry:
    """Registry keeping stable identity separate from mutable provider operations."""

    def __init__(self) -> None:
        self._registrations: dict[str, ProviderRegistration] = {}
        self._operational: dict[str, ProviderOperationalState] = {}

    def register(self, identity: ProviderIdentity) -> None:
        existing = self._registrations.get(identity.provider_id)
        if existing is not None and existing.identity != identity:
            raise ProviderIdentityError(
                f"provider_id {identity.provider_id!r} is already bound to another identity"
            )
        if existing is None:
            self._registrations[identity.provider_id] = ProviderRegistration(identity=identity)
            self._operational[identity.provider_id] = ProviderOperationalState()

    def get(self, provider_id: str) -> ProviderRegistration:
        try:
            return self._registrations[provider_id]
        except KeyError as exc:
            raise ProviderIdentityError(f"unknown provider_id {provider_id!r}") from exc

    def operational_state(self, provider_id: str) -> ProviderOperationalState:
        self.get(provider_id)
        return self._operational[provider_id]

    def set_operational_state(
        self, provider_id: str, state: ProviderOperationalState
    ) -> None:
        self.get(provider_id)
        self._operational[provider_id] = state

    def attach_model(self, provider_id: str, model_id: str) -> None:
        if not model_id.strip():
            raise ProviderIdentityError("model_id must be non-empty")
        registration = self.get(provider_id)
        self._registrations[provider_id] = replace(
            registration, model_ids=registration.model_ids | {model_id}
        )

    def attach_route(self, provider_id: str, route_id: str) -> None:
        if not route_id.strip():
            raise ProviderIdentityError("route_id must be non-empty")
        registration = self.get(provider_id)
        self._registrations[provider_id] = replace(
            registration, route_ids=registration.route_ids | {route_id}
        )

    def registrations(self) -> tuple[ProviderRegistration, ...]:
        return tuple(self._registrations[key] for key in sorted(self._registrations))
