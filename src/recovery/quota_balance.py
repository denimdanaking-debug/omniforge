"""Quota-aware load balancing for provider/route candidates.

This is narrow Phase 6 quota balancing, not full dynamic routing scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.providers.identity import ProviderQuotaState, QuotaSignal


class QuotaPressure(StrEnum):
    """Deterministic quota pressure categories."""

    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"

    def severity(self) -> int:
        """Return numeric severity where lower means preferable for dispatch."""
        return {
            QuotaPressure.LOW: 0,
            QuotaPressure.MODERATE: 1,
            QuotaPressure.HIGH: 2,
            QuotaPressure.CRITICAL: 3,
            QuotaPressure.UNKNOWN: 4,
            QuotaPressure.EXHAUSTED: 5,
        }[self]

    @classmethod
    def from_quota(cls, quota: ProviderQuotaState | None) -> QuotaPressure:
        if quota is None:
            return cls.UNKNOWN
        if quota.is_exhausted():
            return cls.EXHAUSTED

        pressure = quota.effective_pressure()
        if pressure is not None:
            if pressure >= 0.9:
                return cls.CRITICAL
            if pressure >= 0.75:
                return cls.HIGH
            if pressure >= 0.5:
                return cls.MODERATE
            if pressure >= 0.0:
                return cls.LOW

        if quota.provider_signal is QuotaSignal.LIMITED:
            return cls.HIGH
        if quota.provider_signal is QuotaSignal.AVAILABLE:
            return cls.LOW

        return cls.UNKNOWN


@dataclass(frozen=True)
class QuotaCandidate:
    """A candidate route with its quota state for balancing."""

    provider_id: str
    route_id: str
    quota: ProviderQuotaState | None
    quota_domain: str | None = None


class QuotaBalancer:
    """Order candidates by quota pressure, excluding exhausted capacity."""

    def select(
        self,
        candidates: list[QuotaCandidate],
        *,
        quota_domain_states: dict[str, ProviderQuotaState] | None = None,
    ) -> tuple[QuotaCandidate, ...]:
        """Return eligible candidates ordered from lowest to highest quota pressure."""
        domain_states = quota_domain_states or {}
        eligible: list[tuple[QuotaCandidate, QuotaPressure]] = []

        for candidate in candidates:
            # Shared quota domain takes precedence over individual route quota.
            quota: ProviderQuotaState | None
            if candidate.quota_domain and candidate.quota_domain in domain_states:
                quota = domain_states[candidate.quota_domain]
            else:
                quota = candidate.quota

            pressure = QuotaPressure.from_quota(quota)
            if pressure is QuotaPressure.EXHAUSTED:
                continue
            eligible.append((candidate, pressure))

        # Deterministic ordering: lower pressure first, then provider_id, route_id.
        eligible.sort(key=lambda item: (item[1].severity(), item[0].provider_id, item[0].route_id))
        return tuple(candidate for candidate, _ in eligible)

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuotaBalancer:
        return cls()
