"""Structured telemetry events for the recovery engine.

These events feed later dashboard/learning phases; Phase 6 only emits them.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.recovery.clock import Clock, SystemClock


class RecoveryEventType(StrEnum):
    """Canonical recovery event types."""

    STATE_TRANSITION = "state_transition"
    QUOTA_PRESSURE_CHANGE = "quota_pressure_change"
    RECOVERY_SCHEDULED = "recovery_scheduled"
    RECHECK_ATTEMPTED = "recheck_attempted"
    RECHECK_SUCCEEDED = "recheck_succeeded"
    RECHECK_FAILED = "recheck_failed"
    FALLBACK_SELECTED = "fallback_selected"
    WAIT_ENTERED = "wait_entered"
    WAIT_RESUMED = "wait_resumed"
    RESERVE_PROTECTED = "reserve_protected"
    RESERVE_CONSUMED = "reserve_consumed"
    DOMAIN_OUTAGE_DETECTED = "domain_outage_detected"


@dataclass(frozen=True)
class RecoveryEvent:
    """One structured recovery event."""

    event_type: RecoveryEventType
    timestamp: datetime.datetime
    provider_id: str | None = None
    route_id: str | None = None
    failure_domain: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "provider_id": self.provider_id,
            "route_id": self.route_id,
            "failure_domain": self.failure_domain,
            "payload": self.payload,
        }


class RecoveryTelemetryBuffer:
    """Bounded in-memory buffer of recovery events."""

    def __init__(self, limit: int = 1024, clock: Clock | None = None) -> None:
        self._limit = limit
        self._clock = clock or SystemClock()
        self._events: list[RecoveryEvent] = []

    def emit(
        self,
        event_type: RecoveryEventType,
        *,
        provider_id: str | None = None,
        route_id: str | None = None,
        failure_domain: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RecoveryEvent:
        event = RecoveryEvent(
            event_type=event_type,
            timestamp=self._clock.now(),
            provider_id=provider_id,
            route_id=route_id,
            failure_domain=failure_domain,
            payload=payload or {},
        )
        self._events.append(event)
        if len(self._events) > self._limit:
            self._events = self._events[-self._limit :]
        return event

    def events(self) -> tuple[RecoveryEvent, ...]:
        return tuple(self._events)

    def clear(self) -> None:
        self._events.clear()
