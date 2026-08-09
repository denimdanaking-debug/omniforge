"""Outage survival orchestration: dispatch or enter persisted wait."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from src.providers.identity import ProviderQuotaState
from src.recovery.clock import Clock, SystemClock, ensure_aware
from src.recovery.quota_balance import QuotaBalancer, QuotaCandidate
from src.recovery.reserve import ReserveCapacityPolicy, evaluate_reserve_eligibility
from src.recovery.state_machine import RouteRecoveryState
from src.recovery.telemetry import RecoveryEventType, RecoveryTelemetryBuffer
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole


@dataclass(frozen=True)
class DispatchChoice:
    """A chosen provider/model/route for dispatch."""

    provider_id: str
    model_id: str
    route_id: str


@dataclass(frozen=True)
class WaitReason:
    """Reason for entering WAITING_FOR_PROVIDER."""

    reason: str
    affected_failure_domains: frozenset[str]
    next_recheck_at: datetime.datetime | None


@dataclass(frozen=True)
class DispatchDecision:
    """Outcome of dispatch-or-wait evaluation."""

    dispatch: bool
    choice: DispatchChoice | None = None
    wait: WaitReason | None = None


@dataclass
class SurvivalCandidate:
    """One candidate route under consideration by the survival engine."""

    provider_id: str
    model_id: str
    route_id: str
    capabilities: ModelCapabilities
    recovery_state: RouteRecoveryState
    quota: ProviderQuotaState | None = None
    quota_domain: str | None = None


@dataclass
class OutageSurvivalEngine:
    """Decide whether to dispatch to an eligible route or enter persisted wait."""

    clock: Clock = field(default_factory=SystemClock)
    quota_balancer: QuotaBalancer = field(default_factory=QuotaBalancer)
    telemetry: RecoveryTelemetryBuffer | None = None

    def dispatch_or_wait(
        self,
        *,
        role: ExecutionRole,
        candidates: list[SurvivalCandidate],
        reserve_policy: ReserveCapacityPolicy | None = None,
        quota_domain_states: dict[str, ProviderQuotaState] | None = None,
    ) -> DispatchDecision:
        """Return dispatch choice or a persisted wait reason."""
        now = self.clock.now()

        eligible = self._filter_eligible(candidates)
        if not eligible:
            return self._wait(
                "no_eligible_routes",
                candidates,
                now,
            )

        after_reserve = self._apply_reserve_policy(role, eligible, reserve_policy)
        if not after_reserve:
            return self._wait(
                "reserve_capacity_protected",
                candidates,
                now,
            )

        quota_candidates = [
            QuotaCandidate(
                provider_id=c.provider_id,
                route_id=c.route_id,
                quota=c.quota,
                quota_domain=c.quota_domain,
            )
            for c in after_reserve
        ]
        ordered = self.quota_balancer.select(
            quota_candidates,
            quota_domain_states=quota_domain_states,
        )
        if not ordered:
            return self._wait(
                "all_eligible_routes_quota_exhausted",
                candidates,
                now,
            )

        chosen = next(
            c
            for c in after_reserve
            if c.provider_id == ordered[0].provider_id and c.route_id == ordered[0].route_id
        )

        if self.telemetry is not None:
            self.telemetry.emit(
                RecoveryEventType.FALLBACK_SELECTED,
                provider_id=chosen.provider_id,
                route_id=chosen.route_id,
                failure_domain=chosen.recovery_state.failure_domain or None,
                payload={"model_id": chosen.model_id, "role": role.value},
            )

        return DispatchDecision(
            dispatch=True,
            choice=DispatchChoice(
                provider_id=chosen.provider_id,
                model_id=chosen.model_id,
                route_id=chosen.route_id,
            ),
        )

    def _filter_eligible(self, candidates: list[SurvivalCandidate]) -> list[SurvivalCandidate]:
        return [c for c in candidates if c.recovery_state.is_eligible()]

    def _apply_reserve_policy(
        self,
        role: ExecutionRole,
        candidates: list[SurvivalCandidate],
        policy: ReserveCapacityPolicy | None,
    ) -> list[SurvivalCandidate]:
        if policy is None:
            return candidates
        result: list[SurvivalCandidate] = []
        for candidate in candidates:
            reserve_result = evaluate_reserve_eligibility(
                role=role,
                provider_id=candidate.provider_id,
                model_id=candidate.model_id,
                route_id=candidate.route_id,
                quota_state=candidate.quota,
                policy=policy,
            )
            if reserve_result.eligible:
                result.append(candidate)
            elif self.telemetry is not None:
                self.telemetry.emit(
                    RecoveryEventType.RESERVE_PROTECTED,
                    provider_id=candidate.provider_id,
                    route_id=candidate.route_id,
                    payload={"reason": reserve_result.reason, "role": role.value},
                )
        return result

    def _wait(
        self,
        reason: str,
        candidates: list[SurvivalCandidate],
        now: datetime.datetime,
    ) -> DispatchDecision:
        domains: set[str] = set()
        next_rechecks: list[datetime.datetime] = []
        for c in candidates:
            if c.recovery_state.failure_domain:
                domains.add(c.recovery_state.failure_domain)
            if c.recovery_state.next_recheck_at is not None:
                next_rechecks.append(c.recovery_state.next_recheck_at)

        next_recheck_at = min(next_rechecks) if next_rechecks else None
        if next_recheck_at is None:
            next_recheck_at = now + datetime.timedelta(seconds=30)

        wait = WaitReason(
            reason=reason,
            affected_failure_domains=frozenset(domains),
            next_recheck_at=next_recheck_at,
        )

        if self.telemetry is not None:
            self.telemetry.emit(
                RecoveryEventType.WAIT_ENTERED,
                payload={
                    "reason": reason,
                    "affected_failure_domains": sorted(domains),
                    "next_recheck_at": next_recheck_at.isoformat(),
                },
            )

        return DispatchDecision(dispatch=False, wait=wait)


@dataclass
class PersistedWait:
    """A persisted WAITING_FOR_PROVIDER entry."""

    task_id: str
    role: str
    reason: str
    affected_failure_domains: frozenset[str]
    next_recheck_at: datetime.datetime
    attempted_candidates: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        ensure_aware(self.next_recheck_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role,
            "reason": self.reason,
            "affected_failure_domains": sorted(self.affected_failure_domains),
            "next_recheck_at": self.next_recheck_at.isoformat(),
            "attempted_candidates": self.attempted_candidates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistedWait:
        next_recheck_at = datetime.datetime.fromisoformat(data["next_recheck_at"])
        ensure_aware(next_recheck_at)
        return cls(
            task_id=data["task_id"],
            role=data["role"],
            reason=data["reason"],
            affected_failure_domains=frozenset(data.get("affected_failure_domains", [])),
            next_recheck_at=next_recheck_at,
            attempted_candidates=list(data.get("attempted_candidates", [])),
        )
