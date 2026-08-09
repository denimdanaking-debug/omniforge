"""Telemetry captured during context strategy execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.context.schema import AuthorityPresence


@dataclass(frozen=True)
class ContextStrategyTelemetry:
    """Immutable telemetry record for one context build."""

    strategy: str
    packet_id: str
    source_item_count: int
    raw_item_count: int
    summary_count: int
    estimated_input_chars: int
    context_capacity: int | None
    budget_consumed: dict[str, Any]
    excluded_count: int
    authority_presence: AuthorityPresence
    provenance_coverage: float
    truncation_events: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source_item_count < 0:
            raise ValueError("source_item_count must be non-negative")
        if self.raw_item_count < 0:
            raise ValueError("raw_item_count must be non-negative")
        if self.summary_count < 0:
            raise ValueError("summary_count must be non-negative")
        if self.estimated_input_chars < 0:
            raise ValueError("estimated_input_chars must be non-negative")
        if self.excluded_count < 0:
            raise ValueError("excluded_count must be non-negative")
        if not 0.0 <= self.provenance_coverage <= 1.0:
            raise ValueError("provenance_coverage must be between 0.0 and 1.0")
