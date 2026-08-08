"""Tests for the normalized provider health interface (Step 2.5)."""

from __future__ import annotations

import pytest

from providers.contracts.health import HealthStatus, ProviderHealth
from providers.contracts.identity import ProviderId, RouteId


@pytest.mark.parametrize(
    ("status", "available", "needs_wait"),
    [
        (HealthStatus.HEALTHY, True, False),
        (HealthStatus.DEGRADED, True, False),
        (HealthStatus.RATE_LIMITED, False, True),
        (HealthStatus.QUOTA_EXHAUSTED, False, True),
        (HealthStatus.COOLING, False, True),
        (HealthStatus.UNAVAILABLE, False, True),
        (HealthStatus.AUTHENTICATION_FAILED, False, True),
        (HealthStatus.DISABLED, False, False),
        (HealthStatus.UNKNOWN, False, False),
    ],
)
def test_health_status_helpers(status: HealthStatus, available: bool, needs_wait: bool) -> None:
    health = ProviderHealth(status=status)
    assert health.is_available() is available
    assert health.needs_recovery_wait() is needs_wait


def test_health_metadata_fields() -> None:
    health = ProviderHealth(
        status=HealthStatus.RATE_LIMITED,
        provider_id=ProviderId("stub"),
        route_id=RouteId("direct"),
        observed_at="2026-08-08T07:00:00Z",
        reason="rate limit hit",
        next_retry_at="2026-08-08T07:00:30Z",
        reset_at="2026-08-08T08:00:00Z",
        consecutive_failures=3,
        last_success_at="2026-08-08T06:55:00Z",
        failure_domain_id="stub-domain",
        metadata={"region": "us-east"},
    )
    assert health.is_rate_limited() is True
    assert health.is_quota_exhausted() is False
    assert health.observed_at == "2026-08-08T07:00:00Z"
    assert health.reason == "rate limit hit"
    assert health.next_retry_at == "2026-08-08T07:00:30Z"
    assert health.reset_at == "2026-08-08T08:00:00Z"
    assert health.consecutive_failures == 3
    assert health.last_success_at == "2026-08-08T06:55:00Z"
    assert health.failure_domain_id == "stub-domain"


def test_health_consecutive_failures_non_negative() -> None:
    with pytest.raises(ValueError):
        ProviderHealth(status=HealthStatus.HEALTHY, consecutive_failures=-1)
