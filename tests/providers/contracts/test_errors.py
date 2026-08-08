"""Exhaustive tests for the normalized provider error taxonomy (Step 2.4)."""

from __future__ import annotations

import pytest

from providers.contracts.errors import (
    ErrorCategory,
    ProviderError,
    ProviderErrorCode,
)
from providers.contracts.identity import ModelId, ProviderId, RouteId


@pytest.mark.parametrize(
    ("code", "expected_category"),
    [
        (ProviderErrorCode.RATE_LIMITED, ErrorCategory.INFRASTRUCTURE),
        (ProviderErrorCode.PROVIDER_UNAVAILABLE, ErrorCategory.INFRASTRUCTURE),
        (ProviderErrorCode.TRANSIENT_TRANSPORT, ErrorCategory.INFRASTRUCTURE),
        (ProviderErrorCode.QUOTA_EXHAUSTED, ErrorCategory.QUOTA),
        (ProviderErrorCode.AUTH_FAILURE, ErrorCategory.AUTH),
        (ProviderErrorCode.UNSUPPORTED_CAPABILITY, ErrorCategory.CAPABILITY),
        (ProviderErrorCode.CONTEXT_OVERFLOW, ErrorCategory.CAPABILITY),
        (ProviderErrorCode.INVALID_MODEL_OUTPUT, ErrorCategory.MODEL_QUALITY),
        (ProviderErrorCode.TASK_FAILURE, ErrorCategory.PROMPT_CONSTRUCTION),
        (ProviderErrorCode.CANCELLED, ErrorCategory.CANCELLATION),
        (ProviderErrorCode.UNKNOWN, ErrorCategory.UNKNOWN),
    ],
)
def test_error_code_default_category(
    code: ProviderErrorCode, expected_category: ErrorCategory
) -> None:
    error = ProviderError(code=code, message="test")
    assert error.category is expected_category


def test_quota_exhaustion_is_not_model_quality() -> None:
    error = ProviderError(
        code=ProviderErrorCode.QUOTA_EXHAUSTED,
        message="Out of quota",
        provider_id=ProviderId("stub"),
        model_id=ModelId("model"),
    )
    assert error.is_quota() is True
    assert error.is_infrastructure() is False
    assert error.is_model_quality() is False


def test_provider_outage_is_not_model_quality() -> None:
    error = ProviderError(
        code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
        message="Provider down",
        provider_id=ProviderId("stub"),
        model_id=ModelId("model"),
    )
    assert error.is_infrastructure() is True
    assert error.is_model_quality() is False


def test_invalid_model_output_is_model_quality() -> None:
    error = ProviderError(
        code=ProviderErrorCode.INVALID_MODEL_OUTPUT,
        message="Unparseable JSON",
        provider_id=ProviderId("stub"),
        model_id=ModelId("model"),
    )
    assert error.is_model_quality() is True
    assert error.is_infrastructure() is False


def test_error_attribution_fields() -> None:
    error = ProviderError(
        code=ProviderErrorCode.RATE_LIMITED,
        message="Rate limited",
        retryable=True,
        provider_id=ProviderId("stub"),
        model_id=ModelId("model"),
        route_id=RouteId("direct"),
        http_status=429,
        provider_error_code="rate_limit",
        retry_after_seconds=30,
        quota_reset_at="2026-08-08T08:00:00Z",
        safe_diagnostic_message="Too many requests; retry after 30s",
        raw_metadata={"headers": {"retry-after": "30"}},
    )
    assert error.retryable is True
    assert error.provider_id == ProviderId("stub")
    assert error.model_id == ModelId("model")
    assert error.route_id == RouteId("direct")
    assert error.http_status == 429
    assert error.provider_error_code == "rate_limit"
    assert error.retry_after_seconds == 30
    assert error.quota_reset_at == "2026-08-08T08:00:00Z"
    assert error.safe_diagnostic_message == "Too many requests; retry after 30s"
    assert error.raw_metadata == {"headers": {"retry-after": "30"}}


def test_error_requires_message() -> None:
    with pytest.raises(ValueError):
        ProviderError(code=ProviderErrorCode.UNKNOWN, message="")


def test_error_safe_diagnostic_defaults_to_message() -> None:
    error = ProviderError(code=ProviderErrorCode.UNKNOWN, message="boom")
    assert error.safe_diagnostic_message == "boom"
