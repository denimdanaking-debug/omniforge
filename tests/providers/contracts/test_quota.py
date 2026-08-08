"""Tests for the normalized provider quota interface (Step 2.6)."""

from __future__ import annotations

import pytest

from providers.contracts.identity import ProviderId, RouteId
from providers.contracts.quota import ProviderQuota, QuotaSignal


def test_unknown_quota_is_not_unlimited() -> None:
    quota = ProviderQuota()
    assert quota.is_known() is False
    assert quota.is_exhausted() is False
    assert quota.effective_pressure() is None


def test_known_quota_reports_pressure() -> None:
    quota = ProviderQuota(
        provider_id=ProviderId("stub"),
        route_id=RouteId("direct"),
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


def test_partial_quota_only_reports_known_pressures() -> None:
    quota = ProviderQuota(
        remaining_requests=25,
        request_limit=100,
    )
    assert quota.is_known() is True
    assert quota.request_pressure() == 0.75
    assert quota.token_pressure() is None
    assert quota.concurrency_pressure() is None
    assert quota.effective_pressure() == 0.75


def test_exhausted_request_quota() -> None:
    quota = ProviderQuota(remaining_requests=0, request_limit=100)
    assert quota.is_exhausted() is True


def test_exhausted_token_quota() -> None:
    quota = ProviderQuota(remaining_tokens=0, token_limit=1000)
    assert quota.is_exhausted() is True


def test_exhausted_by_signal() -> None:
    quota = ProviderQuota(provider_signal=QuotaSignal.EXHAUSTED)
    assert quota.is_exhausted() is True


def test_concurrency_saturation() -> None:
    quota = ProviderQuota(concurrency_limit=4, active_concurrency=4)
    assert quota.is_exhausted() is True
    assert quota.concurrency_pressure() == 1.0


def test_upcoming_reset() -> None:
    quota = ProviderQuota(
        remaining_requests=10,
        request_limit=100,
        reset_at="2026-08-08T08:00:00Z",
    )
    assert quota.reset_at == "2026-08-08T08:00:00Z"
    assert quota.is_exhausted() is False


def test_negative_quota_values_rejected() -> None:
    with pytest.raises(ValueError):
        ProviderQuota(remaining_requests=-1)
