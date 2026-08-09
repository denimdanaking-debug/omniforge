"""Exhaustive tests for the normalized provider error taxonomy (Step 2.4)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from src.providers.errors import (
    ErrorCategory,
    ProviderError,
    ProviderErrorCode,
)
from src.providers.identity import ProviderIdentity
from src.routing.model_identity import ModelIdentity

SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_errors_12345"


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
        (ProviderErrorCode.TASK_FAILURE, ErrorCategory.TASK),
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
        provider_id=ProviderIdentity("stub", "Stub", "stub.example"),
        model_id=ModelIdentity(model_id="model", family="family"),
    )
    assert error.is_quota() is True
    assert error.is_infrastructure() is False
    assert error.is_model_quality() is False


def test_provider_outage_is_not_model_quality() -> None:
    error = ProviderError(
        code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
        message="Provider down",
        provider_id=ProviderIdentity("stub", "Stub", "stub.example"),
        model_id=ModelIdentity(model_id="model", family="family"),
    )
    assert error.is_infrastructure() is True
    assert error.is_model_quality() is False


def test_invalid_model_output_is_model_quality() -> None:
    error = ProviderError(
        code=ProviderErrorCode.INVALID_MODEL_OUTPUT,
        message="Unparseable JSON",
        provider_id=ProviderIdentity("stub", "Stub", "stub.example"),
        model_id=ModelIdentity(model_id="model", family="family"),
    )
    assert error.is_model_quality() is True
    assert error.is_infrastructure() is False


def test_task_failure_is_neutral_attribution() -> None:
    error = ProviderError(
        code=ProviderErrorCode.TASK_FAILURE,
        message="Task could not be completed",
        provider_id=ProviderIdentity("stub", "Stub", "stub.example"),
        model_id=ModelIdentity(model_id="model", family="family"),
    )
    assert error.category is ErrorCategory.TASK
    assert error.is_model_quality() is False
    assert error.is_prompt_construction() is False
    assert error.is_infrastructure() is False


def test_error_attribution_fields() -> None:
    error = ProviderError(
        code=ProviderErrorCode.RATE_LIMITED,
        message="Rate limited",
        retryable=True,
        provider_id=ProviderIdentity("stub", "Stub", "stub.example"),
        model_id=ModelIdentity(model_id="model", family="family"),
        http_status=429,
        provider_error_code="rate_limit",
        retry_after_seconds=30,
        quota_reset_at="2026-08-08T08:00:00Z",
        safe_diagnostic_message="Too many requests; retry after 30s",
        raw_metadata={"headers": {"retry-after": "30"}},
    )
    assert error.retryable is True
    assert error.provider_id == ProviderIdentity("stub", "Stub", "stub.example")
    assert error.model_id == ModelIdentity(model_id="model", family="family")
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


def test_error_message_is_redacted_at_construction() -> None:
    error = ProviderError(
        code=ProviderErrorCode.AUTH_FAILURE,
        message=f"provider failed password={SENTINEL}",
    )
    assert error.safe_diagnostic_message is not None
    assert SENTINEL not in error.message
    assert SENTINEL not in error.safe_diagnostic_message


def test_error_message_key_aware_redaction_replaces_entire_value() -> None:
    error = ProviderError(
        code=ProviderErrorCode.AUTH_FAILURE,
        message="provider failed",
        raw_metadata={"password": SENTINEL, "safe": "keep"},
    )
    assert error.raw_metadata["password"] == "<redacted>"
    assert error.raw_metadata["safe"] == "keep"
    assert SENTINEL not in json.dumps(dataclasses.asdict(error), default=str)


def test_error_safe_diagnostic_message_is_redacted_even_when_supplied() -> None:
    error = ProviderError(
        code=ProviderErrorCode.AUTH_FAILURE,
        message="provider failed",
        safe_diagnostic_message=f"auth failed password={SENTINEL}",
    )
    assert error.safe_diagnostic_message is not None
    assert SENTINEL not in error.safe_diagnostic_message


def test_error_raw_metadata_is_key_aware_redacted() -> None:
    error = ProviderError(
        code=ProviderErrorCode.AUTH_FAILURE,
        message="provider failed",
        raw_metadata={"password": SENTINEL, "other": "ok"},
    )
    assert error.raw_metadata["password"] == "<redacted>"
    assert error.raw_metadata["other"] == "ok"
    assert SENTINEL not in json.dumps(dataclasses.asdict(error), default=str)


def test_error_repr_does_not_leak_sentinel() -> None:
    error = ProviderError(
        code=ProviderErrorCode.AUTH_FAILURE,
        message=f"provider failed password={SENTINEL}",
        raw_metadata={"password": SENTINEL},
    )
    representation = repr(error)
    assert SENTINEL not in representation
    assert "<redacted>" in representation


def test_error_str_does_not_leak_sentinel() -> None:
    error = ProviderError(
        code=ProviderErrorCode.AUTH_FAILURE,
        message=f"provider failed password={SENTINEL}",
    )
    assert SENTINEL not in str(error)
