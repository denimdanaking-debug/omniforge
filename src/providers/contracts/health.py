"""Normalized provider health interface.

Health is reported by provider adapters and consumed by the router. It is
intentionally designed to support future active and passive health checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from providers.contracts.identity import ProviderId, RouteId


class HealthStatus(Enum):
    """Normalized provider health states."""

    HEALTHY = auto()
    DEGRADED = auto()
    RATE_LIMITED = auto()
    QUOTA_EXHAUSTED = auto()
    COOLING = auto()
    UNAVAILABLE = auto()
    AUTHENTICATION_FAILED = auto()
    DISABLED = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Normalized health information for a provider or route.

    Not every provider exposes every field; missing values remain ``None``
    rather than being fabricated.
    """

    status: HealthStatus
    provider_id: ProviderId | None = None
    route_id: RouteId | None = None
    observed_at: str | None = None
    reason: str | None = None
    next_retry_at: str | None = None
    reset_at: str | None = None
    consecutive_failures: int | None = None
    last_success_at: str | None = None
    failure_domain_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.consecutive_failures is not None and self.consecutive_failures < 0:
            raise ValueError("consecutive_failures must be non-negative")

    def is_available(self) -> bool:
        """Return True if the provider/route can be used for dispatch."""
        return self.status in {
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
        }

    def is_rate_limited(self) -> bool:
        """Return True if the provider/route is currently rate limited."""
        return self.status is HealthStatus.RATE_LIMITED

    def is_quota_exhausted(self) -> bool:
        """Return True if the provider/route has exhausted quota."""
        return self.status is HealthStatus.QUOTA_EXHAUSTED

    def needs_recovery_wait(self) -> bool:
        """Return True if dispatch should wait for a retry/reset time."""
        return self.status in {
            HealthStatus.RATE_LIMITED,
            HealthStatus.QUOTA_EXHAUSTED,
            HealthStatus.COOLING,
            HealthStatus.UNAVAILABLE,
            HealthStatus.AUTHENTICATION_FAILED,
        }
