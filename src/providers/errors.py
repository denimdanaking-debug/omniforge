"""Normalized provider error taxonomy.

The taxonomy explicitly distinguishes infrastructure/provider problems, route
problems, model-quality problems, and system/prompt construction problems.
Provider outage or quota exhaustion must never be classified as evidence that
the model itself is bad at coding/review/planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.providers.identity import ProviderIdentity
from src.routing.inference_route import InferenceRouteIdentity
from src.routing.model_identity import ModelIdentity


class ErrorCategory(Enum):
    """High-level error category for retry/routing decisions."""

    INFRASTRUCTURE = auto()
    ROUTE = auto()
    QUOTA = auto()
    AUTH = auto()
    CAPABILITY = auto()
    MODEL_QUALITY = auto()
    PROMPT_CONSTRUCTION = auto()
    TASK = auto()
    CANCELLATION = auto()
    UNKNOWN = auto()


class ProviderErrorCode(Enum):
    """Normalized provider error codes."""

    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_FAILURE = "AUTH_FAILURE"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    TRANSIENT_TRANSPORT = "TRANSIENT_TRANSPORT"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    TASK_FAILURE = "TASK_FAILURE"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


_INFRASTRUCTURE_CODES = {
    ProviderErrorCode.RATE_LIMITED,
    ProviderErrorCode.PROVIDER_UNAVAILABLE,
    ProviderErrorCode.TRANSIENT_TRANSPORT,
}

_QUOTA_CODES = {ProviderErrorCode.QUOTA_EXHAUSTED}

_AUTH_CODES = {ProviderErrorCode.AUTH_FAILURE}

_CAPABILITY_CODES = {
    ProviderErrorCode.UNSUPPORTED_CAPABILITY,
    ProviderErrorCode.CONTEXT_OVERFLOW,
}

_MODEL_QUALITY_CODES = {ProviderErrorCode.INVALID_MODEL_OUTPUT}

_PROMPT_CODES: set[ProviderErrorCode] = set()

_TASK_CODES = {ProviderErrorCode.TASK_FAILURE}

_CANCELLATION_CODES = {ProviderErrorCode.CANCELLED}


@dataclass(frozen=True)
class ProviderError:
    """Normalized provider error with enough context for retry/routing logic.

    Raw provider metadata is captured only when safe and useful. Secrets and
    credentials must never be included.
    """

    code: ProviderErrorCode
    message: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    retryable: bool = False
    provider_id: ProviderIdentity | None = None
    model_id: ModelIdentity | None = None
    route_id: InferenceRouteIdentity | None = None
    http_status: int | None = None
    provider_error_code: str | None = None
    retry_after_seconds: int | None = None
    quota_reset_at: str | None = None
    safe_diagnostic_message: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("ProviderError.message is required")
        if self.category is ErrorCategory.UNKNOWN:
            object.__setattr__(self, "category", _category_for_code(self.code))
        if self.safe_diagnostic_message is None:
            object.__setattr__(self, "safe_diagnostic_message", self.message)

    def is_infrastructure(self) -> bool:
        """Return True if this is an infrastructure/provider problem."""
        return self.category in {ErrorCategory.INFRASTRUCTURE, ErrorCategory.ROUTE}

    def is_quota(self) -> bool:
        """Return True if this is a quota/capacity problem."""
        return self.category is ErrorCategory.QUOTA

    def is_auth(self) -> bool:
        """Return True if this is an auth problem."""
        return self.category is ErrorCategory.AUTH

    def is_model_quality(self) -> bool:
        """Return True if this reflects on model output quality."""
        return self.category is ErrorCategory.MODEL_QUALITY

    def is_prompt_construction(self) -> bool:
        """Return True if this reflects a prompt/system construction problem."""
        return self.category is ErrorCategory.PROMPT_CONSTRUCTION


def _category_for_code(code: ProviderErrorCode) -> ErrorCategory:
    """Derive the default error category from a normalized error code."""
    if code in _INFRASTRUCTURE_CODES:
        return ErrorCategory.INFRASTRUCTURE
    if code in _QUOTA_CODES:
        return ErrorCategory.QUOTA
    if code in _AUTH_CODES:
        return ErrorCategory.AUTH
    if code in _CAPABILITY_CODES:
        return ErrorCategory.CAPABILITY
    if code in _MODEL_QUALITY_CODES:
        return ErrorCategory.MODEL_QUALITY
    if code in _PROMPT_CODES:
        return ErrorCategory.PROMPT_CONSTRUCTION
    if code in _TASK_CODES:
        return ErrorCategory.TASK
    if code in _CANCELLATION_CODES:
        return ErrorCategory.CANCELLATION
    return ErrorCategory.UNKNOWN
