"""Deterministic provider/route health state machine.

Provider availability is not model quality. Infrastructure signals affect route
and provider operational state only.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from src.providers.errors import ProviderErrorCode
from src.providers.identity import ProviderHealth
from src.recovery.backoff import BackoffPolicy, HotLoopPolicy, RetryBudget
from src.recovery.clock import Clock, isoformat, parse_iso
from src.recovery.scheduler import RecheckPolicy, RecoveryScheduler
from src.recovery.signals import ProviderSignal, SignalKind


@dataclass(frozen=True)
class HealthTransition:
    """One recorded health-state transition."""

    prior: ProviderHealth
    new: ProviderHealth
    observed_at: datetime.datetime
    reason: str
    source_signal: ProviderSignal
    retry_after: datetime.datetime | None = None
    reset_at: datetime.datetime | None = None


@dataclass(frozen=True)
class RouteRecoveryState:
    """Immutable recovery state for one provider/route.

    An unobserved route starts as ``DEGRADED``: usable only as a conservative
    fallback until positive success or health-check evidence transitions it to
    ``HEALTHY``. No adapter object or persisted record is assumed healthy by
    default.
    """

    health: ProviderHealth = ProviderHealth.DEGRADED
    consecutive_failures: int = 0
    last_success_at: datetime.datetime | None = None
    last_failure_at: datetime.datetime | None = None
    cooldown_until: datetime.datetime | None = None
    next_recheck_at: datetime.datetime | None = None
    quota_reset_at: datetime.datetime | None = None
    failure_domain: str = ""
    reason: str | None = "unobserved"
    transition_log: tuple[HealthTransition, ...] = ()

    def __post_init__(self) -> None:
        if self.consecutive_failures < 0:
            raise ValueError("consecutive_failures must be non-negative")
        for name, value in (
            ("last_success_at", self.last_success_at),
            ("last_failure_at", self.last_failure_at),
            ("cooldown_until", self.cooldown_until),
            ("next_recheck_at", self.next_recheck_at),
            ("quota_reset_at", self.quota_reset_at),
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")

    def is_eligible(self) -> bool:
        return self.health in {ProviderHealth.HEALTHY, ProviderHealth.DEGRADED}

    def needs_recovery_wait(self) -> bool:
        return self.health in {
            ProviderHealth.RATE_LIMITED,
            ProviderHealth.QUOTA_EXHAUSTED,
            ProviderHealth.COOLING,
            ProviderHealth.UNAVAILABLE,
            ProviderHealth.AUTH_FAILED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "health": self.health.value,
            "consecutive_failures": self.consecutive_failures,
            "last_success_at": isoformat(self.last_success_at) if self.last_success_at else None,
            "last_failure_at": isoformat(self.last_failure_at) if self.last_failure_at else None,
            "cooldown_until": isoformat(self.cooldown_until) if self.cooldown_until else None,
            "next_recheck_at": isoformat(self.next_recheck_at) if self.next_recheck_at else None,
            "quota_reset_at": isoformat(self.quota_reset_at) if self.quota_reset_at else None,
            "failure_domain": self.failure_domain,
            "reason": self.reason,
            "transition_log": [
                {
                    "prior": t.prior.value,
                    "new": t.new.value,
                    "observed_at": isoformat(t.observed_at),
                    "reason": t.reason,
                    "retry_after": isoformat(t.retry_after) if t.retry_after else None,
                    "reset_at": isoformat(t.reset_at) if t.reset_at else None,
                    "source_provider_id": t.source_signal.provider_id,
                    "source_route_id": t.source_signal.route_id,
                    "source_failure_domain": t.source_signal.failure_domain,
                    "source_kind": t.source_signal.kind.value,
                }
                for t in self.transition_log
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouteRecoveryState:
        def _parse(name: str) -> datetime.datetime | None:
            value = data.get(name)
            return parse_iso(value) if value else None

        def _source_signal(t: dict[str, Any], observed_at: datetime.datetime) -> ProviderSignal:
            kind_value = t.get("source_kind", SignalKind.ERROR.value)
            return ProviderSignal(
                provider_id=t.get("source_provider_id") or "unknown",
                route_id=t.get("source_route_id"),
                failure_domain=t.get("source_failure_domain") or "unknown",
                timestamp=observed_at,
                kind=SignalKind(kind_value),
            )

        log = data.get("transition_log") or []
        transitions = tuple(
            HealthTransition(
                prior=ProviderHealth(t["prior"]),
                new=ProviderHealth(t["new"]),
                observed_at=parse_iso(t["observed_at"]),
                reason=t["reason"],
                source_signal=_source_signal(t, parse_iso(t["observed_at"])),
                retry_after=_parse_from_value(t.get("retry_after")),
                reset_at=_parse_from_value(t.get("reset_at")),
            )
            for t in log
        )
        return cls(
            health=ProviderHealth(data["health"]),
            consecutive_failures=data.get("consecutive_failures", 0),
            last_success_at=_parse("last_success_at"),
            last_failure_at=_parse("last_failure_at"),
            cooldown_until=_parse("cooldown_until"),
            next_recheck_at=_parse("next_recheck_at"),
            quota_reset_at=_parse("quota_reset_at"),
            failure_domain=data.get("failure_domain", ""),
            reason=data.get("reason"),
            transition_log=transitions,
        )


def _parse_from_value(value: str | None) -> datetime.datetime | None:
    return parse_iso(value) if value else None


@dataclass(frozen=True)
class StateMachineConfig:
    """Configuration for the health state machine."""

    recheck_policy: RecheckPolicy = field(default_factory=RecheckPolicy)
    backoff_policy: BackoffPolicy = field(default_factory=BackoffPolicy)
    retry_budget: RetryBudget = field(default_factory=RetryBudget)
    hot_loop_policy: HotLoopPolicy = field(default_factory=HotLoopPolicy)
    transition_log_limit: int = 16


class HealthStateMachine:
    """Apply normalized provider signals to route recovery state deterministically."""

    def __init__(self, clock: Clock, config: StateMachineConfig | None = None) -> None:
        self._clock = clock
        self._config = config or StateMachineConfig()
        self._scheduler = RecoveryScheduler(policy=self._config.recheck_policy)

    @property
    def clock(self) -> Clock:
        return self._clock

    def apply(
        self,
        state: RouteRecoveryState,
        signal: ProviderSignal,
    ) -> RouteRecoveryState:
        """Return a new recovery state after applying the signal."""
        if signal.kind is SignalKind.SUCCESS:
            return self._apply_success(state, signal)
        if signal.kind is SignalKind.ERROR:
            return self._apply_error(state, signal)
        if signal.kind is SignalKind.QUOTA:
            return self._apply_quota(state, signal)
        if signal.kind is SignalKind.HEALTH_CHECK:
            return self._apply_health_check(state, signal)
        return state

    def _apply_success(
        self,
        state: RouteRecoveryState,
        signal: ProviderSignal,
    ) -> RouteRecoveryState:
        now = self._clock.now()
        health = state.health

        if health is ProviderHealth.DISABLED:
            return state

        if health is ProviderHealth.AUTH_FAILED:
            # Credential failures require explicit admin/config-changed signal.
            return state

        if health in {
            ProviderHealth.HEALTHY,
            ProviderHealth.DEGRADED,
            ProviderHealth.RATE_LIMITED,
            ProviderHealth.QUOTA_EXHAUSTED,
            ProviderHealth.COOLING,
            ProviderHealth.UNAVAILABLE,
        }:
            # A success while cooling/unavailable/rate-limited is treated as a
            # successful recheck that restores eligibility.
            return self._transition_to(
                state,
                ProviderHealth.HEALTHY,
                signal,
                reason="success/recovered",
                last_success_at=now,
                reset_failures=True,
            )

        return state

    def _apply_error(
        self,
        state: RouteRecoveryState,
        signal: ProviderSignal,
    ) -> RouteRecoveryState:
        if state.health is ProviderHealth.DISABLED:
            return state

        error = signal.error
        if error is None:
            return state

        now = self._clock.now()
        code = error.code

        # Errors that must never affect provider health.
        if code in {
            ProviderErrorCode.UNSUPPORTED_CAPABILITY,
            ProviderErrorCode.CONTEXT_OVERFLOW,
            ProviderErrorCode.INVALID_MODEL_OUTPUT,
            ProviderErrorCode.TASK_FAILURE,
            ProviderErrorCode.CANCELLED,
        }:
            return state

        consecutive = state.consecutive_failures + 1

        if code is ProviderErrorCode.RATE_LIMITED:
            retry_after = self._parse_retry_after(error.retry_after_seconds, now)
            return self._transition_to(
                state,
                ProviderHealth.RATE_LIMITED,
                signal,
                reason=f"rate_limited: {error.message}",
                last_failure_at=now,
                cooldown_until=retry_after,
                next_recheck_at=retry_after,
                retry_after=retry_after,
                increment_failures=True,
            )

        if code is ProviderErrorCode.QUOTA_EXHAUSTED:
            reset_at = self._parse_reset_at(error.quota_reset_at, now)
            next_check = reset_at or (
                now
                + datetime.timedelta(
                    seconds=self._config.recheck_policy.quota_exhausted_unknown_reset_seconds
                )
            )
            return self._transition_to(
                state,
                ProviderHealth.QUOTA_EXHAUSTED,
                signal,
                reason=f"quota_exhausted: {error.message}",
                last_failure_at=now,
                quota_reset_at=reset_at,
                next_recheck_at=next_check,
                reset_at=reset_at,
                increment_failures=True,
            )

        if code is ProviderErrorCode.AUTH_FAILURE:
            next_check = now + datetime.timedelta(
                seconds=self._config.recheck_policy.auth_failed_recheck_seconds
            )
            return self._transition_to(
                state,
                ProviderHealth.AUTH_FAILED,
                signal,
                reason=f"auth_failed: {error.message}",
                last_failure_at=now,
                next_recheck_at=next_check,
                increment_failures=True,
            )

        if code is ProviderErrorCode.PROVIDER_UNAVAILABLE:
            next_check = now + datetime.timedelta(
                seconds=self._config.backoff_policy.delay_for_attempt(consecutive - 1)
            )
            return self._transition_to(
                state,
                ProviderHealth.UNAVAILABLE,
                signal,
                reason=f"unavailable: {error.message}",
                last_failure_at=now,
                next_recheck_at=next_check,
                increment_failures=True,
            )

        if code is ProviderErrorCode.TRANSIENT_TRANSPORT:
            if self._config.retry_budget.exceeds_consecutive(consecutive):
                target = ProviderHealth.UNAVAILABLE
                reason = "transient_transport exceeded retry budget"
                next_check = now + datetime.timedelta(
                    seconds=self._config.backoff_policy.delay_for_attempt(consecutive - 1)
                )
            else:
                target = ProviderHealth.DEGRADED
                reason = f"transient_transport: {error.message}"
                next_check = now + datetime.timedelta(
                    seconds=self._config.recheck_policy.degraded_recheck_seconds
                )
            return self._transition_to(
                state,
                target,
                signal,
                reason=reason,
                last_failure_at=now,
                next_recheck_at=next_check,
                increment_failures=True,
            )

        if code is ProviderErrorCode.UNKNOWN:
            return self._transition_to(
                state,
                ProviderHealth.DEGRADED,
                signal,
                reason=f"unknown_error: {error.message}",
                last_failure_at=now,
                next_recheck_at=now
                + datetime.timedelta(seconds=self._config.recheck_policy.degraded_recheck_seconds),
                increment_failures=True,
            )

        return state

    def _apply_quota(
        self,
        state: RouteRecoveryState,
        signal: ProviderSignal,
    ) -> RouteRecoveryState:
        if state.health is ProviderHealth.DISABLED:
            return state

        quota = signal.quota
        if quota is None:
            return state

        now = self._clock.now()

        if quota.is_exhausted():
            reset_at = self._parse_reset_at(quota.reset_at, now)
            next_check = reset_at or (
                now
                + datetime.timedelta(
                    seconds=self._config.recheck_policy.quota_exhausted_unknown_reset_seconds
                )
            )
            return self._transition_to(
                state,
                ProviderHealth.QUOTA_EXHAUSTED,
                signal,
                reason="quota signal exhausted",
                quota_reset_at=reset_at,
                next_recheck_at=next_check,
                reset_at=reset_at,
            )

        # Non-exhausted quota is capacity telemetry, not provider health degradation.
        # If the route was previously quota-exhausted, this verified non-exhausted
        # report is evidence of recovery.
        if state.health is ProviderHealth.QUOTA_EXHAUSTED:
            return self._transition_to(
                state,
                ProviderHealth.HEALTHY,
                signal,
                reason="quota_recovered",
                last_success_at=now,
                reset_failures=True,
            )

        return state

    def _apply_health_check(
        self,
        state: RouteRecoveryState,
        signal: ProviderSignal,
    ) -> RouteRecoveryState:
        if state.health is ProviderHealth.DISABLED:
            return state

        operational = signal.operational_state
        if operational is None:
            return state

        reported = operational.health
        now = self._clock.now()

        # A health check returning HEALTHY is evidence of recovery.
        if reported is ProviderHealth.HEALTHY and state.health in {
            ProviderHealth.DEGRADED,
            ProviderHealth.COOLING,
            ProviderHealth.UNAVAILABLE,
            ProviderHealth.RATE_LIMITED,
            ProviderHealth.QUOTA_EXHAUSTED,
        }:
            return self._transition_to(
                state,
                ProviderHealth.HEALTHY,
                signal,
                reason="health_check_recovered",
                last_success_at=now,
                reset_failures=True,
            )

        # A health check reporting a problem transitions to that state.
        if reported in {
            ProviderHealth.DEGRADED,
            ProviderHealth.RATE_LIMITED,
            ProviderHealth.QUOTA_EXHAUSTED,
            ProviderHealth.UNAVAILABLE,
            ProviderHealth.AUTH_FAILED,
        }:
            next_check = self._scheduler.compute_next_recheck(
                RouteRecoveryState(health=reported, failure_domain=signal.failure_domain),
                self._clock,
            )
            return self._transition_to(
                state,
                reported,
                signal,
                reason=f"health_check_reported_{reported.value}",
                last_failure_at=now,
                next_recheck_at=next_check,
            )

        return state

    def _transition_to(
        self,
        state: RouteRecoveryState,
        new_health: ProviderHealth,
        signal: ProviderSignal,
        reason: str,
        *,
        last_success_at: datetime.datetime | None = None,
        last_failure_at: datetime.datetime | None = None,
        cooldown_until: datetime.datetime | None = None,
        next_recheck_at: datetime.datetime | None = None,
        quota_reset_at: datetime.datetime | None = None,
        retry_after: datetime.datetime | None = None,
        reset_at: datetime.datetime | None = None,
        increment_failures: bool = False,
        reset_failures: bool = False,
    ) -> RouteRecoveryState:
        prior = state.health
        if reset_failures:
            consecutive = 0
        else:
            consecutive = state.consecutive_failures + (1 if increment_failures else 0)

        transition = HealthTransition(
            prior=prior,
            new=new_health,
            observed_at=signal.timestamp,
            reason=reason,
            source_signal=signal,
            retry_after=retry_after,
            reset_at=reset_at,
        )
        log = state.transition_log + (transition,)
        limit = self._config.transition_log_limit
        if len(log) > limit:
            log = log[-limit:]

        return RouteRecoveryState(
            health=new_health,
            consecutive_failures=consecutive,
            last_success_at=last_success_at or state.last_success_at,
            last_failure_at=last_failure_at or state.last_failure_at,
            cooldown_until=cooldown_until,
            next_recheck_at=next_recheck_at,
            quota_reset_at=quota_reset_at,
            failure_domain=signal.failure_domain or state.failure_domain,
            reason=reason,
            transition_log=log,
        )

    def _parse_retry_after(
        self,
        seconds: int | None,
        now: datetime.datetime,
    ) -> datetime.datetime | None:
        if seconds is None or seconds <= 0:
            base = self._config.recheck_policy.rate_limited_base_seconds
            return now + datetime.timedelta(seconds=base)
        return now + datetime.timedelta(seconds=seconds)

    def _parse_reset_at(
        self,
        value: str | None,
        now: datetime.datetime,
    ) -> datetime.datetime | None:
        if value is None:
            return None
        try:
            parsed = datetime.datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return None
            return parsed if parsed > now else None
        except ValueError:
            return None
