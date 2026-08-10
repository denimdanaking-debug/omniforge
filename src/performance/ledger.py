"""Append-only performance-event ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from src.performance.event import PerformanceEvent
from src.security.redaction import redact


@dataclass(frozen=True)
class PerformanceLedger:
    """Immutable source-of-truth ledger of performance events.

    Events are append-only. Corrections are represented as new events, never by
    mutating existing records. Duplicate events (by deterministic event_id) are
    rejected so replay after restart is idempotent.
    """

    events: tuple[PerformanceEvent, ...] = ()
    schema_version: str = "1.0.0"

    def append(self, event: PerformanceEvent) -> PerformanceLedger:
        """Return a new ledger with ``event`` appended.

        Raises:
            ValueError: if an event with the same ``event_id`` is already present.
        """
        if any(e.event_id == event.event_id for e in self.events):
            raise ValueError(f"duplicate event_id: {event.event_id}")
        return PerformanceLedger(
            events=self.events + (event,),
            schema_version=self.schema_version,
        )

    def append_all(self, events: tuple[PerformanceEvent, ...]) -> PerformanceLedger:
        """Return a new ledger with multiple events appended atomically.

        Raises:
            ValueError: if any event_id collides with an existing event or another
            event in the batch.
        """
        seen: set[str] = set()
        for event in self.events:
            seen.add(event.event_id)
        for event in events:
            if event.event_id in seen:
                raise ValueError(f"duplicate event_id: {event.event_id}")
            seen.add(event.event_id)
        return PerformanceLedger(
            events=self.events + events,
            schema_version=self.schema_version,
        )

    def has_event(self, event_id: str) -> bool:
        return any(e.event_id == event_id for e in self.events)

    def events_for_task(self, task_id: str) -> tuple[PerformanceEvent, ...]:
        return tuple(e for e in self.events if e.task_id == task_id)

    def events_for_model(self, model_id: str) -> tuple[PerformanceEvent, ...]:
        return tuple(e for e in self.events if e.model_id == model_id)

    def events_for_role(self, role: str) -> tuple[PerformanceEvent, ...]:
        return tuple(e for e in self.events if e.execution_role == role)

    def events_for_project(self, project_id: str) -> tuple[PerformanceEvent, ...]:
        return tuple(e for e in self.events if e.project_id == project_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "events": [e.to_safe_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceLedger:
        events = tuple(
            PerformanceEvent.from_dict(event_data) for event_data in data.get("events", [])
        )
        return cls(
            events=events,
            schema_version=str(data.get("schema_version", "1.0.0")),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a fully redacted dict safe for filesystem persistence."""
        return cast(dict[str, Any], redact(self.to_dict()))
