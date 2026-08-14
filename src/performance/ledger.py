"""Append-only performance-event ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from src.performance.event import PerformanceEvent, performance_event_fingerprint
from src.security.redaction import redact


@dataclass(frozen=True)
class PerformanceLedger:
    """Immutable source-of-truth ledger of performance events.

    Events are append-only. Corrections are represented as new events, never by
    mutating existing records. Replay is idempotent: the same ``event_id`` with
    the same canonical fingerprint is a no-op, while the same ``event_id`` with
    a different fingerprint fails closed.
    """

    events: tuple[PerformanceEvent, ...] = ()
    schema_version: str = "1.0.0"

    def _verify_event_fingerprint(self, event: PerformanceEvent) -> None:
        expected = performance_event_fingerprint(event)
        if event.event_fingerprint != expected:
            raise ValueError(
                f"event fingerprint mismatch for {event.event_id}: "
                f"stored={event.event_fingerprint}, expected={expected}"
            )

    def append(self, event: PerformanceEvent) -> PerformanceLedger:
        """Return a new ledger with ``event`` appended.

        Exact replay (same ``event_id`` and same fingerprint) returns the
        unchanged ledger.  A fingerprint collision fails closed.

        Raises:
            ValueError: if the event fingerprint is forged/stale or if the same
            ``event_id`` exists with a different fingerprint.
        """
        self._verify_event_fingerprint(event)
        existing_by_id = {e.event_id: e for e in self.events}
        existing = existing_by_id.get(event.event_id)
        if existing is not None:
            if existing.event_fingerprint == event.event_fingerprint:
                return self
            raise ValueError(
                f"event_id collision for {event.event_id}: "
                f"existing fingerprint differs from appended event"
            )
        return PerformanceLedger(
            events=self.events + (event,),
            schema_version=self.schema_version,
        )

    def append_all(self, events: tuple[PerformanceEvent, ...]) -> PerformanceLedger:
        """Return a new ledger with multiple events appended atomically.

        Validates every fingerprint and detects collisions (against existing
        events and within the batch) before appending.  Exact duplicates are
        idempotent; conflicting duplicates fail the entire batch.

        Raises:
            ValueError: if any fingerprint is forged/stale or if any event_id
            collides with a different fingerprint.
        """
        existing_by_id = {e.event_id: e for e in self.events}
        new_events: list[PerformanceEvent] = []
        batch_seen: dict[str, PerformanceEvent] = {}

        for event in events:
            self._verify_event_fingerprint(event)
            existing = existing_by_id.get(event.event_id)
            if existing is not None:
                if existing.event_fingerprint == event.event_fingerprint:
                    continue
                raise ValueError(
                    f"event_id collision for {event.event_id}: "
                    f"existing fingerprint differs from batch event"
                )
            prior_batch = batch_seen.get(event.event_id)
            if prior_batch is not None:
                if prior_batch.event_fingerprint == event.event_fingerprint:
                    continue
                raise ValueError(
                    f"event_id collision within batch for {event.event_id}: "
                    f"conflicting fingerprints"
                )
            batch_seen[event.event_id] = event
            new_events.append(event)

        return PerformanceLedger(
            events=self.events + tuple(new_events),
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
        events: list[PerformanceEvent] = []
        seen_ids: set[str] = set()
        for event_data in data.get("events", []):
            event = PerformanceEvent.from_dict(event_data)
            if event.event_id in seen_ids:
                raise ValueError(f"duplicate event_id in ledger: {event.event_id}")
            seen_ids.add(event.event_id)
            expected = performance_event_fingerprint(event)
            if event.event_fingerprint != expected:
                raise ValueError(
                    f"event fingerprint mismatch for {event.event_id}: "
                    f"stored={event.event_fingerprint}, expected={expected}"
                )
            events.append(event)
        return cls(
            events=tuple(events),
            schema_version=str(data.get("schema_version", "1.0.0")),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a fully redacted dict safe for filesystem persistence."""
        return cast(dict[str, Any], redact(self.to_dict()))
