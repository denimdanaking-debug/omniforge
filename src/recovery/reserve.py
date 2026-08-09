"""Reserve-capacity policy to protect premium capacity for critical roles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.providers.identity import ProviderQuotaState
from src.recovery.quota_balance import QuotaPressure
from src.routing.roles import ExecutionRole


@dataclass(frozen=True)
class ReserveCapacityPolicy:
    """Policy controlling which roles may consume reserved provider/model/route capacity."""

    reserved_provider_ids: frozenset[str] = field(default_factory=frozenset)
    reserved_model_ids: frozenset[str] = field(default_factory=frozenset)
    reserved_route_ids: frozenset[str] = field(default_factory=frozenset)
    reserved_roles: frozenset[str] = field(default_factory=frozenset)
    minimum_remaining_fraction: float | None = None
    emergency_override_roles: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.minimum_remaining_fraction is not None and not (
            0.0 <= self.minimum_remaining_fraction <= 1.0
        ):
            raise ValueError("minimum_remaining_fraction must be between 0.0 and 1.0")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReserveCapacityPolicy:
        return cls(
            reserved_provider_ids=frozenset(data.get("reserved_provider_ids", [])),
            reserved_model_ids=frozenset(data.get("reserved_model_ids", [])),
            reserved_route_ids=frozenset(data.get("reserved_route_ids", [])),
            reserved_roles=frozenset(data.get("reserved_roles", [])),
            minimum_remaining_fraction=data.get("minimum_remaining_fraction"),
            emergency_override_roles=frozenset(data.get("emergency_override_roles", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "reserved_provider_ids": sorted(self.reserved_provider_ids),
            "reserved_model_ids": sorted(self.reserved_model_ids),
            "reserved_route_ids": sorted(self.reserved_route_ids),
            "reserved_roles": sorted(self.reserved_roles),
            "emergency_override_roles": sorted(self.emergency_override_roles),
        }
        if self.minimum_remaining_fraction is not None:
            result["minimum_remaining_fraction"] = self.minimum_remaining_fraction
        return result


@dataclass(frozen=True)
class ReserveEligibilityResult:
    eligible: bool
    reason: str


def evaluate_reserve_eligibility(
    *,
    role: ExecutionRole,
    provider_id: str,
    model_id: str,
    route_id: str,
    quota_state: ProviderQuotaState | None,
    policy: ReserveCapacityPolicy | None,
) -> ReserveEligibilityResult:
    """Determine whether a request role may consume a candidate's capacity."""
    if policy is None:
        return ReserveEligibilityResult(True, "no_reserve_policy")

    is_reserved = (
        provider_id in policy.reserved_provider_ids
        or model_id in policy.reserved_model_ids
        or route_id in policy.reserved_route_ids
    )
    if not is_reserved:
        return ReserveEligibilityResult(True, "candidate_not_reserved")

    role_value = role.value
    if role_value in policy.reserved_roles:
        return ReserveEligibilityResult(True, "role_allowed_in_reserve")

    if role_value in policy.emergency_override_roles:
        return ReserveEligibilityResult(True, "emergency_override")

    # Even reserved candidates can be used if quota pressure is low enough
    # and no minimum reserve floor is breached.
    if policy.minimum_remaining_fraction is not None and quota_state is not None:
        pressure = QuotaPressure.from_quota(quota_state)
        if pressure in {QuotaPressure.UNKNOWN, QuotaPressure.LOW}:
            return ReserveEligibilityResult(True, "low_pressure_allows_reserve_use")
        remaining = quota_state.remaining_fraction
        if remaining is not None and remaining >= policy.minimum_remaining_fraction:
            return ReserveEligibilityResult(True, "remaining_fraction_sufficient")

    return ReserveEligibilityResult(
        False,
        "reserve_capacity_protected_for_critical_roles",
    )
