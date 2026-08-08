"""Normalized provider quota/capacity reporting.

Missing quota data is never treated as unlimited capacity. Pressure calculations
are only performed when enough information exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from providers.contracts.identity import ProviderId, RouteId


class QuotaSignal(Enum):
    """Provider-defined capacity signal."""

    AVAILABLE = auto()
    LIMITED = auto()
    EXHAUSTED = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class ProviderQuota:
    """Normalized quota/capacity report.

    ``None`` means explicitly unknown/unreported, not unlimited.
    """

    provider_id: ProviderId | None = None
    route_id: RouteId | None = None
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    request_limit: int | None = None
    token_limit: int | None = None
    reset_at: str | None = None
    concurrency_limit: int | None = None
    active_concurrency: int | None = None
    provider_signal: QuotaSignal = QuotaSignal.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        non_negative = {
            "remaining_requests": self.remaining_requests,
            "remaining_tokens": self.remaining_tokens,
            "request_limit": self.request_limit,
            "token_limit": self.token_limit,
            "concurrency_limit": self.concurrency_limit,
            "active_concurrency": self.active_concurrency,
        }
        for name, value in non_negative.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def is_known(self) -> bool:
        """Return True if any concrete quota information is present."""
        return any(
            value is not None
            for value in (
                self.remaining_requests,
                self.remaining_tokens,
                self.request_limit,
                self.token_limit,
                self.concurrency_limit,
                self.active_concurrency,
            )
        )

    def is_exhausted(self) -> bool:
        """Return True if quota is known to be exhausted.

        Missing data is not treated as unlimited; this returns True only when a
        provider explicitly reports zero remaining capacity or an EXHAUSTED signal.
        """
        return (
            self.provider_signal is QuotaSignal.EXHAUSTED
            or self.remaining_requests == 0
            or self.remaining_tokens == 0
            or (
                self.concurrency_limit is not None
                and self.active_concurrency is not None
                and self.concurrency_limit > 0
                and self.active_concurrency >= self.concurrency_limit
            )
        )

    def request_pressure(self) -> float | None:
        """Return request pressure ratio [0.0, 1.0] when both limit and usage are known."""
        if self.request_limit is None or self.remaining_requests is None:
            return None
        if self.request_limit <= 0:
            return None
        used = self.request_limit - self.remaining_requests
        return used / self.request_limit

    def token_pressure(self) -> float | None:
        """Return token pressure ratio [0.0, 1.0] when both limit and usage are known."""
        if self.token_limit is None or self.remaining_tokens is None:
            return None
        if self.token_limit <= 0:
            return None
        used = self.token_limit - self.remaining_tokens
        return used / self.token_limit

    def concurrency_pressure(self) -> float | None:
        """Return concurrency saturation ratio [0.0, 1.0] when both values are known."""
        if self.concurrency_limit is None or self.active_concurrency is None:
            return None
        if self.concurrency_limit <= 0:
            return None
        return self.active_concurrency / self.concurrency_limit

    def effective_pressure(self) -> float | None:
        """Return the highest known pressure ratio, or None if no ratio can be computed."""
        pressures = [
            self.request_pressure(),
            self.token_pressure(),
            self.concurrency_pressure(),
        ]
        known = [p for p in pressures if p is not None]
        return max(known) if known else None
