"""Centralized provider signal ingestion for the recovery engine.

Adapters report normalized signals. The recovery engine interprets them.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.providers.errors import ProviderError
from src.providers.identity import ProviderOperationalState, ProviderQuotaState
from src.providers.response import ProviderResponse
from src.recovery.clock import Clock, SystemClock


class SignalKind(StrEnum):
    """Kind of provider signal."""

    SUCCESS = "success"
    ERROR = "error"
    QUOTA = "quota"
    HEALTH_CHECK = "health_check"


@dataclass(frozen=True)
class ProviderSignal:
    """Normalized observation used by the health state machine."""

    provider_id: str
    route_id: str | None
    failure_domain: str
    timestamp: datetime.datetime
    kind: SignalKind
    error: ProviderError | None = None
    quota: ProviderQuotaState | None = None
    operational_state: ProviderOperationalState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("provider_id must be non-empty")
        if not self.failure_domain.strip():
            raise ValueError("failure_domain must be non-empty")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")


def signal_from_response(
    response: ProviderResponse,
    *,
    route_id: str | None,
    failure_domain: str,
    clock: Clock | None = None,
) -> ProviderSignal:
    """Create a success signal from a normalized provider response."""
    now = (clock or SystemClock()).now()
    return ProviderSignal(
        provider_id=response.provider_id.provider_id,
        route_id=route_id,
        failure_domain=failure_domain,
        timestamp=now,
        kind=SignalKind.SUCCESS,
        metadata={"request_id": response.request_id},
    )


def signal_from_error(
    error: ProviderError,
    *,
    route_id: str | None,
    failure_domain: str,
    clock: Clock | None = None,
) -> ProviderSignal:
    """Create an error signal from a normalized provider error."""
    now = (clock or SystemClock()).now()
    provider_id = error.provider_id.provider_id if error.provider_id else "unknown"
    return ProviderSignal(
        provider_id=provider_id,
        route_id=route_id,
        failure_domain=failure_domain,
        timestamp=now,
        kind=SignalKind.ERROR,
        error=error,
    )


def signal_from_quota(
    quota: ProviderQuotaState,
    *,
    provider_id: str,
    route_id: str | None,
    failure_domain: str,
    clock: Clock | None = None,
) -> ProviderSignal:
    """Create a quota signal from a normalized quota state."""
    now = (clock or SystemClock()).now()
    return ProviderSignal(
        provider_id=provider_id,
        route_id=route_id,
        failure_domain=failure_domain,
        timestamp=now,
        kind=SignalKind.QUOTA,
        quota=quota,
    )


def signal_from_health_check(
    state: ProviderOperationalState,
    *,
    provider_id: str,
    route_id: str | None,
    failure_domain: str,
    clock: Clock | None = None,
) -> ProviderSignal:
    """Create a health-check signal from an adapter health() result."""
    now = (clock or SystemClock()).now()
    return ProviderSignal(
        provider_id=provider_id,
        route_id=route_id,
        failure_domain=failure_domain,
        timestamp=now,
        kind=SignalKind.HEALTH_CHECK,
        operational_state=state,
    )
