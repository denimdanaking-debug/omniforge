"""Tests for normalized provider quota/health extensions (Steps 2.5, 2.6)."""

from __future__ import annotations

import pytest

from src.providers.identity import (
    ProviderHealth,
    ProviderOperationalState,
    ProviderQuotaState,
    QuotaSignal,
)


class TestQuotaState:
    def test_unknown_quota_is_not_unlimited(self) -> None:
        quota = ProviderQuotaState()
        assert quota.is_known() is False
        assert quota.is_exhausted() is False
        assert quota.effective_pressure() is None

    def test_known_quota_reports_pressure(self) -> None:
        quota = ProviderQuotaState(
            remaining_requests=50,
            request_limit=100,
            remaining_tokens=500,
            token_limit=1000,
            concurrency_limit=10,
            active_concurrency=5,
        )
        assert quota.is_known() is True
        assert quota.is_exhausted() is False
        assert quota.request_pressure() == 0.5
        assert quota.token_pressure() == 0.5
        assert quota.concurrency_pressure() == 0.5
        assert quota.effective_pressure() == 0.5

    def test_partial_quota_only_reports_known_pressures(self) -> None:
        quota = ProviderQuotaState(remaining_requests=25, request_limit=100)
        assert quota.is_known() is True
        assert quota.request_pressure() == 0.75
        assert quota.token_pressure() is None
        assert quota.concurrency_pressure() is None
        assert quota.effective_pressure() == 0.75

    def test_exhausted_request_quota(self) -> None:
        quota = ProviderQuotaState(remaining_requests=0, request_limit=100)
        assert quota.is_exhausted() is True

    def test_exhausted_token_quota(self) -> None:
        quota = ProviderQuotaState(remaining_tokens=0, token_limit=1000)
        assert quota.is_exhausted() is True

    def test_exhausted_by_signal(self) -> None:
        quota = ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED)
        assert quota.is_exhausted() is True

    def test_concurrency_saturation(self) -> None:
        quota = ProviderQuotaState(concurrency_limit=4, active_concurrency=4)
        assert quota.is_exhausted() is True
        assert quota.concurrency_pressure() == 1.0

    def test_upcoming_reset(self) -> None:
        quota = ProviderQuotaState(
            remaining_requests=10,
            request_limit=100,
            reset_at="2026-08-08T08:00:00Z",
        )
        assert quota.reset_at == "2026-08-08T08:00:00Z"
        assert quota.is_exhausted() is False

    def test_negative_quota_values_rejected(self) -> None:
        with pytest.raises(ValueError):
            ProviderQuotaState(remaining_requests=-1)

    @pytest.mark.parametrize(
        ("remaining_fraction", "expected_pressure"),
        [
            (0.90, 0.10),
            (0.10, 0.90),
            (0.0, 1.0),
            (1.0, 0.0),
        ],
    )
    def test_remaining_fraction_pressure(
        self, remaining_fraction: float, expected_pressure: float
    ) -> None:
        quota = ProviderQuotaState(remaining_fraction=remaining_fraction)
        assert quota.effective_pressure() == pytest.approx(expected_pressure)

    def test_effective_pressure_uses_highest_actual_pressure(self) -> None:
        quota = ProviderQuotaState(
            remaining_fraction=0.90,  # 10% pressure
            remaining_requests=10,
            request_limit=100,  # 90% pressure
            remaining_tokens=500,
            token_limit=1000,  # 50% pressure
            concurrency_limit=10,
            active_concurrency=2,  # 20% pressure
        )
        assert quota.effective_pressure() == 0.90

    def test_remaining_requests_cannot_exceed_limit(self) -> None:
        with pytest.raises(ValueError):
            ProviderQuotaState(remaining_requests=101, request_limit=100)

    def test_remaining_tokens_cannot_exceed_limit(self) -> None:
        with pytest.raises(ValueError):
            ProviderQuotaState(remaining_tokens=1001, token_limit=1000)

    def test_active_concurrency_cannot_exceed_limit(self) -> None:
        with pytest.raises(ValueError):
            ProviderQuotaState(active_concurrency=5, concurrency_limit=4)


class TestOperationalState:
    @pytest.mark.parametrize(
        ("status", "available", "needs_wait"),
        [
            (ProviderHealth.HEALTHY, True, False),
            (ProviderHealth.DEGRADED, True, False),
            (ProviderHealth.RATE_LIMITED, False, True),
            (ProviderHealth.QUOTA_EXHAUSTED, False, True),
            (ProviderHealth.COOLING, False, True),
            (ProviderHealth.UNAVAILABLE, False, True),
            (ProviderHealth.AUTH_FAILED, False, True),
            (ProviderHealth.DISABLED, False, False),
        ],
    )
    def test_health_status_helpers(
        self, status: ProviderHealth, available: bool, needs_wait: bool
    ) -> None:
        health = ProviderOperationalState(health=status)
        assert health.is_available() is available
        assert health.needs_recovery_wait() is needs_wait

    def test_metadata_fields(self) -> None:
        health = ProviderOperationalState(
            health=ProviderHealth.RATE_LIMITED,
            last_checked_at="2026-08-08T07:00:00Z",
            reason="rate limit hit",
            next_retry_at="2026-08-08T07:00:30Z",
            reset_at="2026-08-08T08:00:00Z",
            consecutive_failures=3,
            last_success_at="2026-08-08T06:55:00Z",
            failure_domain_id="stub-domain",
        )
        assert health.is_rate_limited() is True
        assert health.is_quota_exhausted() is False
        assert health.last_checked_at == "2026-08-08T07:00:00Z"
        assert health.reason == "rate limit hit"
        assert health.next_retry_at == "2026-08-08T07:00:30Z"
        assert health.reset_at == "2026-08-08T08:00:00Z"
        assert health.consecutive_failures == 3
        assert health.last_success_at == "2026-08-08T06:55:00Z"
        assert health.failure_domain_id == "stub-domain"

    def test_consecutive_failures_non_negative(self) -> None:
        with pytest.raises(ValueError):
            ProviderOperationalState(consecutive_failures=-1)
