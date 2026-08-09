"""Deterministic recovery recheck scheduler."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.providers.identity import ProviderHealth
from src.recovery.clock import Clock

if TYPE_CHECKING:
    from src.recovery.state_machine import RouteRecoveryState


@dataclass(frozen=True)
class RecheckPolicy:
    """Policy controlling when routes are due for recovery rechecks."""

    rate_limited_base_seconds: float = 30.0
    quota_exhausted_unknown_reset_seconds: float = 300.0
    unavailable_backoff: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0, 120.0, 300.0)
    unavailable_max_seconds: float = 600.0
    degraded_recheck_seconds: float = 60.0
    auth_failed_recheck_seconds: float = 300.0
    cooling_recheck_seconds: float = 30.0

    def unavailable_delay(self, attempt: int) -> float:
        if attempt < 0:
            attempt = 0
        if attempt < len(self.unavailable_backoff):
            return min(self.unavailable_backoff[attempt], self.unavailable_max_seconds)
        return min(self.unavailable_backoff[-1], self.unavailable_max_seconds)


@dataclass
class RecoveryScheduler:
    """In-memory queue of routes due for recheck.

    No network calls. No sleeps. Time is evaluated through the injected clock.
    """

    policy: RecheckPolicy = field(default_factory=RecheckPolicy)
    _scheduled: dict[str, datetime.datetime] = field(default_factory=dict)

    def schedule(self, route_id: str, next_recheck_at: datetime.datetime) -> None:
        if next_recheck_at.tzinfo is None:
            raise ValueError("next_recheck_at must be timezone-aware")
        self._scheduled[route_id] = next_recheck_at

    def unschedule(self, route_id: str) -> None:
        self._scheduled.pop(route_id, None)

    def due_routes(self, clock: Clock) -> tuple[str, ...]:
        now = clock.now()
        return tuple(
            route_id for route_id in sorted(self._scheduled) if self._scheduled[route_id] <= now
        )

    def pop_due(self, clock: Clock) -> tuple[str, ...]:
        due = self.due_routes(clock)
        for route_id in due:
            self._scheduled.pop(route_id, None)
        return due

    def next_due_at(self, route_id: str) -> datetime.datetime | None:
        return self._scheduled.get(route_id)

    def compute_next_recheck(
        self,
        state: RouteRecoveryState,
        clock: Clock,
    ) -> datetime.datetime:
        """Compute next recheck time from recovery state and policy."""
        now = clock.now()
        health = state.health

        if health is ProviderHealth.RATE_LIMITED:
            seconds = self.policy.rate_limited_base_seconds
            if state.cooldown_until is not None and state.cooldown_until > now:
                return max(now + datetime.timedelta(seconds=seconds), state.cooldown_until)
            return now + datetime.timedelta(seconds=seconds)

        if health is ProviderHealth.QUOTA_EXHAUSTED:
            if state.quota_reset_at is not None and state.quota_reset_at > now:
                return state.quota_reset_at
            seconds = self.policy.quota_exhausted_unknown_reset_seconds
            return now + datetime.timedelta(seconds=seconds)

        if health is ProviderHealth.UNAVAILABLE:
            attempt = max(0, state.consecutive_failures)
            delay = self.policy.unavailable_delay(attempt)
            return now + datetime.timedelta(seconds=delay)

        if health is ProviderHealth.AUTH_FAILED:
            return now + datetime.timedelta(seconds=self.policy.auth_failed_recheck_seconds)

        if health is ProviderHealth.DEGRADED:
            return now + datetime.timedelta(seconds=self.policy.degraded_recheck_seconds)

        if health is ProviderHealth.COOLING:
            if state.cooldown_until is not None and state.cooldown_until > now:
                return state.cooldown_until
            return now + datetime.timedelta(seconds=self.policy.cooling_recheck_seconds)

        # HEALTHY or DISABLED: no recheck needed.
        return now
