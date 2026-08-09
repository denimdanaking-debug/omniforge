"""Injectable time source for deterministic recovery logic and tests."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Protocol


class Clock(Protocol):
    """Provider-neutral clock; production uses system time, tests use a fixed or manual clock."""

    def now(self) -> datetime.datetime:
        """Return a timezone-aware UTC datetime."""
        ...


class SystemClock:
    """Production clock backed by system UTC."""

    def now(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.UTC)


@dataclass
class FixedClock:
    """A clock that always returns the same injected timestamp."""

    timestamp: datetime.datetime = field(default_factory=lambda: _utc_now())

    def now(self) -> datetime.datetime:
        return self.timestamp

    def advance(self, seconds: float) -> datetime.datetime:
        self.timestamp += datetime.timedelta(seconds=seconds)
        return self.timestamp


@dataclass
class ManualClock:
    """A clock whose time is advanced explicitly by tests."""

    _now: datetime.datetime = field(default_factory=lambda: _utc_now())

    def now(self) -> datetime.datetime:
        return self._now

    def set(self, timestamp: datetime.datetime) -> None:
        if timestamp.tzinfo is None:
            raise ValueError("clock timestamp must be timezone-aware")
        self._now = timestamp

    def advance(self, seconds: float) -> datetime.datetime:
        self._now += datetime.timedelta(seconds=seconds)
        return self._now


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def ensure_aware(timestamp: datetime.datetime) -> datetime.datetime:
    """Reject naive datetimes; return aware datetimes unchanged."""
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return timestamp


def isoformat(timestamp: datetime.datetime) -> str:
    """Deterministic ISO-8601 representation for persisted state."""
    ensure_aware(timestamp)
    return timestamp.isoformat()


def parse_iso(value: str) -> datetime.datetime:
    """Parse an ISO-8601 string, rejecting naive values."""
    parsed = datetime.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("parsed timestamp must be timezone-aware")
    return parsed
