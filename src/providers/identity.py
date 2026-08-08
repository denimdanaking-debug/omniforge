"""First-class provider identity and operational state."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

_PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


class ProviderIdentityError(ValueError):
    pass


class QuotaSignal(StrEnum):
    """Provider-defined coarse capacity signal."""

    AVAILABLE = "available"
    LIMITED = "limited"
    EXHAUSTED = "exhausted"
    UNKNOWN = "unknown"


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
    """Normalized quota/capacity report.

    ``None`` means explicitly unknown/unreported, not unlimited.
    """

    remaining_fraction: float | None = None
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    request_limit: int | None = None
    token_limit: int | None = None
    reset_at: str | None = None
    concurrency_limit: int | None = None
    active_concurrency: int | None = None
    provider_signal: QuotaSignal = QuotaSignal.UNKNOWN
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.remaining_fraction is not None and not 0.0 <= self.remaining_fraction <= 1.0:
            raise ProviderIdentityError("remaining_fraction must be between 0 and 1")
        for name, value in (
            ("remaining_requests", self.remaining_requests),
            ("remaining_tokens", self.remaining_tokens),
            ("request_limit", self.request_limit),
            ("token_limit", self.token_limit),
            ("concurrency_limit", self.concurrency_limit),
            ("active_concurrency", self.active_concurrency),
        ):
            if value is not None and value < 0:
                raise ProviderIdentityError(f"{name} must be non-negative")
        for remaining_name, limit_name, remaining, limit in (
            ("remaining_requests", "request_limit", self.remaining_requests, self.request_limit),
            ("remaining_tokens", "token_limit", self.remaining_tokens, self.token_limit),
            (
                "active_concurrency",
                "concurrency_limit",
                self.active_concurrency,
                self.concurrency_limit,
            ),
        ):
            if remaining is not None and limit is not None and remaining > limit:
                raise ProviderIdentityError(
                    f"{remaining_name} ({remaining}) cannot exceed {limit_name} ({limit})"
                )

    def is_known(self) -> bool:
        """Return True if any concrete quota information is present."""
        return any(
            value is not None
            for value in (
                self.remaining_fraction,
                self.remaining_requests,
                self.remaining_tokens,
                self.request_limit,
                self.token_limit,
                self.concurrency_limit,
                self.active_concurrency,
            )
        )

    def is_exhausted(self) -> bool:
        """Return True if quota is known to be exhausted."""
        if self.provider_signal is QuotaSignal.EXHAUSTED:
            return True
        if self.remaining_requests == 0:
            return True
        if self.remaining_tokens == 0:
            return True
        if self.remaining_fraction is not None and self.remaining_fraction <= 0.0:
            return True
        return (
            self.concurrency_limit is not None
            and self.active_concurrency is not None
            and self.concurrency_limit > 0
            and self.active_concurrency >= self.concurrency_limit
        )

    def request_pressure(self) -> float | None:
        """Return request pressure ratio [0.0, 1.0] when limit and usage are known."""
        if self.request_limit is None or self.remaining_requests is None:
            return None
        if self.request_limit <= 0:
            return None
        return (self.request_limit - self.remaining_requests) / self.request_limit

    def token_pressure(self) -> float | None:
        """Return token pressure ratio [0.0, 1.0] when limit and usage are known."""
        if self.token_limit is None or self.remaining_tokens is None:
            return None
        if self.token_limit <= 0:
            return None
        return (self.token_limit - self.remaining_tokens) / self.token_limit

    def concurrency_pressure(self) -> float | None:
        """Return concurrency saturation ratio [0.0, 1.0] when both values are known."""
        if self.concurrency_limit is None or self.active_concurrency is None:
            return None
        if self.concurrency_limit <= 0:
            return None
        return self.active_concurrency / self.concurrency_limit

    def _fraction_pressure(self) -> float | None:
        """Return pressure implied by remaining_fraction (0.0 = full, 1.0 = exhausted)."""
        if self.remaining_fraction is None:
            return None
        return 1.0 - self.remaining_fraction

    def effective_pressure(self) -> float | None:
        """Return the highest known pressure ratio, or None if none can be computed."""
        pressures = [
            self._fraction_pressure(),
            self.request_pressure(),
            self.token_pressure(),
            self.concurrency_pressure(),
        ]
        known = [p for p in pressures if p is not None]
        return max(known) if known else None


@dataclass(frozen=True)
class ProviderOperationalState:
    """Normalized provider/route operational health.

    Missing fields remain ``None`` rather than being fabricated.
    """

    health: ProviderHealth = ProviderHealth.HEALTHY
    quota: ProviderQuotaState = field(default_factory=ProviderQuotaState)
    last_checked_at: str | None = None
    next_retry_at: str | None = None
    reason: str | None = None
    reset_at: str | None = None
    consecutive_failures: int | None = None
    last_success_at: str | None = None
    failure_domain_id: str | None = None

    def __post_init__(self) -> None:
        if self.consecutive_failures is not None and self.consecutive_failures < 0:
            raise ProviderIdentityError("consecutive_failures must be non-negative")

    def is_available(self) -> bool:
        """Return True if the provider/route can be used for dispatch."""
        return self.health in {
            ProviderHealth.HEALTHY,
            ProviderHealth.DEGRADED,
        }

    def is_rate_limited(self) -> bool:
        """Return True if the provider/route is currently rate limited."""
        return self.health is ProviderHealth.RATE_LIMITED

    def is_quota_exhausted(self) -> bool:
        """Return True if the provider/route has exhausted quota."""
        return self.health is ProviderHealth.QUOTA_EXHAUSTED

    def needs_recovery_wait(self) -> bool:
        """Return True if dispatch should wait for a retry/reset time."""
        return self.health in {
            ProviderHealth.RATE_LIMITED,
            ProviderHealth.QUOTA_EXHAUSTED,
            ProviderHealth.COOLING,
            ProviderHealth.UNAVAILABLE,
            ProviderHealth.AUTH_FAILED,
        }


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

    def set_operational_state(self, provider_id: str, state: ProviderOperationalState) -> None:
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
